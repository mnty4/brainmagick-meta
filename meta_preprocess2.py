import pickle
import json
from sklearn.model_selection import train_test_split
from collections import defaultdict
from mne.time_frequency import Spectrum
from sklearn.model_selection import ShuffleSplit
import pandas as pd
from mne.io import Raw
import mne
import numpy as np
from pathlib import Path
import time
from torch.utils.data import Dataset, DataLoader
from omegaconf import OmegaConf
from dora import hydra_main
import logging
import typing as tp
import torch
from hydra import initialize, compose
import hydra
import os
from functools import lru_cache
import functools
from tqdm import tqdm
import torch.utils
import torch.utils.data
from setup_logging import configure_logging
# from . import env
# from .cache import Cache
from bm.dataset import _extract_recordings, _preload, assign_blocks, SegmentDataset
from bm.train import override_args_
from speech_embedding_2 import SpeechEmbeddings
from frozendict import frozendict
from _env import Env as env  # we need this here otherwise submitit pickle does crazy stuff.
from dora.log import LogProgress
import sys
from joblib import Parallel, delayed
from joblib import wrap_non_picklable_objects
import threading
from typing import Tuple
from concurrent.futures import ThreadPoolExecutor


# import transformers
# transformers.logging.set_verbosity_error()

# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# print("SCRIPT_DIR",SCRIPT_DIR)
# sys.path.append(os.path.dirname(SCRIPT_DIR))
Debug= True
logger = logging.getLogger(__name__)
# all print and logging to a file

# mne.set_log_level('ERROR') # shut up mne
base = "/projects/SilSpeech/Dev/SilentSpeech_Se2/listen_meg_eeg_preprocess/brainmagick/bm/" #= os.path.abspath

def list_to_tuple(function: tp.Callable) -> tp.Any:
    """Custom decorator function, to convert list to a tuple."""

    def wrapper(*args, **kwargs) -> tp.Any:
        args = tuple(tuple(x) if isinstance(x, list) else x for x in args)
        kwargs = {k: tuple(v) if isinstance(v, list) else v for k, v in kwargs.items()}
        result = function(*args, **kwargs)
        result = tuple(result) if isinstance(result, list) else result
        return result

    return wrapper

def deep_freeze(thing):
    from collections.abc import Collection, Mapping, Hashable
    from frozendict import frozendict
    if thing is None or isinstance(thing, str):
        return thing
    elif isinstance(thing, Mapping):
        return frozendict({k: deep_freeze(v) for k, v in thing.items()})
    elif isinstance(thing, Collection):
        return tuple(deep_freeze(i) for i in thing)
    elif not isinstance(thing, Hashable):
        raise TypeError(f"unfreezable type: '{type(thing)}'")
    else:
        return thing


def deep_freeze_args(func):
    import functools

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        return func(*deep_freeze(args), **deep_freeze(kwargs))
    return wrapped


def _load_and_preprocess(recording, sample_rate, highpass):
    return recording.preprocessed(sample_rate, highpass).load_data()
# @list_to_tuple
# @deep_freeze_args
# @lru_cache(typed=True)
def get_raw_events(selections: tp.List[tp.Dict[str, tp.Any]],
        n_recordings: int,
        test_ratio: float,
        valid_ratio: float,
        sample_rate: int,  # FIXME
        highpass: float = 0,
        num_workers: int = 10,
        apply_baseline: bool = True,
        progress: bool = False,
        skip_recordings: int = 0,
        min_block_duration: float = 0.0,
        force_uid_assignement: bool = True,
        shuffle_recordings_seed: int = 42,
        split_assign_seed: int = 12,
        min_n_blocks_per_split: int = 20,
        features: tp.Optional[tp.List[str]] = None,
        extra_test_features: tp.Optional[tp.List[str]] = None,
        test: dict = {},
        allow_empty_split: bool = False,
        n_subjects: tp.Optional[int] = None,
        n_subjects_test: tp.Optional[int] = None,
        remove_ratio: float = 0.,
        **factory_kwargs: tp.Any):
    # get from running gwilliams study
    logger.info(f'Loading recordings...')
    all_recordings = _extract_recordings(selections, n_recordings, skip_recordings=skip_recordings,
    shuffle_recordings_seed=shuffle_recordings_seed)
    print("all_recordings",type(all_recordings),len(all_recordings))
    # for ii, recording in enumerate(all_recordings):
    #     print('recording',ii,'subject',recording._subject_index)
    #     print('recording',ii,'session',recording.session)
    #     print('recording',ii,'story',recording.story)

    # reduce the number of recordings for debugging
    if Debug:
        print('Debugging mode, reducing the number of recordings to 16')
        # all_recordings = all_recordings[:16]
        all_recordings = all_recordings[0:8]
    # all_recordings = all_recordings[:2]
    recording_info = all_recordings.copy()

    all_recordings = LogProgress(logger, all_recordings, name="Preparing cache", level=logging.DEBUG)
    print('_preload')
    # all_recordings = [_preload(s, sample_rate=sample_rate, highpass=highpass) for s in all_recordings]
    # run this in parallel
    all_recordings = Parallel(n_jobs=32)(delayed(_preload)(s, sample_rate=sample_rate, highpass=highpass) for s in all_recordings)
    print('done _preload')
    # print(all_recordings)
    # raws = [recording.preprocessed(128, 0).load_data() for recording in all_recordings]
    # convrert to parrallel
    raws = Parallel(n_jobs=32)(delayed(_load_and_preprocess)(recording, sample_rate = 128, highpass = 0) for recording in all_recordings)
    # raws = raw_preloaded_list
    events = [recording for recording in all_recordings]
    # events = [recording.events() for recording in all_recordings]
    infos = [recording.mne_info for recording in all_recordings]
    logger.info(f'Recordings loaded succesfully.')
    return raws, events, infos, recording_info



def apply_baseline(raw, data):
    baseline_start, baseline_end = 0, 0.5

    baseline_idx = np.logical_and(raw.times >= baseline_start, raw.times <= baseline_end)

    baseline_mean = data[:, baseline_idx].mean(axis=1, keepdims=True)

    clamped_data = data - baseline_mean
    return clamped_data

def normalise(data):
    std = np.std(data, axis=1, keepdims=True)
    mean = np.mean(data, axis=1, keepdims=True)
    normalised = (data - mean) / std
    return normalised

def apply_clamp(data):

    mean = data.mean(axis=1, keepdims=True)
    std_dev = data.std(axis=1, keepdims=True)
    min_val = mean - 20 * std_dev
    max_val = mean + 20 * std_dev
    clamped_data = np.clip(data, min_val, max_val)

    return clamped_data

def apply_noise(data):
    return np.random.normal(loc=0, scale=1, size=data.shape)


def _prep_raws(raw, random_noise=False):
    data = raw.get_data()
    if random_noise:
        data = apply_noise(data)
    else:
        data = apply_baseline(raw, data)
        data = apply_clamp(data)
        data = normalise(data)
    return data
        
def preprocess_raws_parallel(raws, random_noise=False, **kwargs):
    prep_results = Parallel(n_jobs=32)(delayed(_prep_raws)(raw, random_noise=random_noise) for raw in raws)
    for raw, data in zip(raws, prep_results):
        raw._data = data

# inplace
def preprocess_raws(raws, random_noise=False, **kwargs):
    raw 
    for raw in raws:
        data = raw.get_data()
        if random_noise:
            data = apply_noise(data)
        else:
            data = apply_baseline(raw, data)
            data = apply_clamp(data)
            data = normalise(data)
        raw._data = data   

def split_dataset(subs: dict, seed, by_trial=False, **kwargs):
    subs_list = subs.values()
    subs_ids = subs.keys()
    task_t = np.array(subs_list)
    # if by_trial:
    #     task_t = task_t.flatten(0, 1)

    tmp_subs, tmp_sub_ids, test_subs, test_sub_ids = train_test_split(task_t, subs_ids, test_size=0.2, random_state=seed, shuffle=True)
    train_subs, train_sub_ids, valid_subs, valid_sub_ids = train_test_split(tmp_subs, tmp_sub_ids, test_size=0.125, random_state=seed, shuffle=True)
    
    # return train, valid, test
    

def _extract( raw, event, info, word_index, offset, n_fft, audio_embedding_length,  subs, segment_lengths, durations, word_count):    
    generate_embeddings = SpeechEmbeddings()
    raw: Raw
    event: pd.DataFrame    
    word_events: pd.DataFrame = event[event['kind'] == 'word']
    # raw.annotations.to_data_frame().info()
    # descs = [json.loads(desc.replace("'", "\"")) for desc in raw.annotations.description]
    # starts = [desc['start'] for desc in descs if desc['kind'] == 'word']
    sub_id = info['subject_info']['his_id']    
    story_id = float(word_events.iloc[0]['story_uid'])
    x = []
    y = []
    w_lbs = []
    skipped = 0
    for i, word_event in word_events.iterrows():    
        raw_start, word_start, duration, word_label, wav_path = word_event['start'], word_event['audio_start'], word_event['duration'], word_event['word'], word_event['sound'] 
        wav_path = wav_path.lower()
        wav_path = os.path.join('/projects/SilSpeech/Dev/SilentSpeech_Se2/listen_meg_eeg_preprocess/brainmagick/data/gwilliams2022_newdl/download/', wav_path)
        raw_start = raw_start + offset
        raw_end = raw_start + duration
        t_idxs = raw.time_as_index([raw_start, raw_end])
        data, times = raw[:, t_idxs[0]:t_idxs[1]] 
        segment_lengths.append(data.shape[-1])
        durations.append(duration)
        if data.shape[-1] <= n_fft // 4:
            skipped += 1
            continue
        #     # pad with zeros if too short for window
        #     to_pad = n_fft - data.shape[-1]
        #     data = np.pad(data, ((0, 0), (0, to_pad)), "constant")
        #     # continue        
        spectrums: Spectrum = raw.compute_psd(tmin=raw_start, tmax=raw_end, fmax=60, n_jobs=32,
                                                n_fft=n_fft, verbose=False, n_per_seg=n_fft // 2, n_overlap=n_fft // 4)
        # spectrums.plot(picks="data", exclude="bads", amplitude=False)
        #     spectrums.plot_topomap(bands = {'Delta (0-4 Hz)': (0, 4), 'Theta (4-8 Hz)': (4, 8),
        #  'Alpha (8-12 Hz)': (8, 12), 'Beta (12-30 Hz)': (12, 30),
        #  'Gamma (30-45 Hz)': (30, 45)}, normalize=True)
        data, freqs = spectrums.get_data(return_freqs=True)
        audio_embedding = generate_embeddings.get_audio_embeddings(wav_path, word_start, duration,
                                                                    audio_embedding_length=audio_embedding_length)
        if word_label in word_index:
            word_id = word_index[word_label]
        else:
            word_id = word_count
            word_index[word_label] = word_count
            word_count += 1
        # print(f'word: {word_label}, audio features: {torch.tensor(audio_label).shape}, PSD: {torch.tensor(data).shape}')
        x.append(data)
        y.append(audio_embedding)
        w_lbs.append(word_id)
    x = torch.tensor(np.array(x)).to(torch.float32)
    y = torch.stack(y).to(torch.float32)
    w_lbs = torch.tensor(w_lbs).to(torch.int64)
    trial = {
        'story_uid': story_id,
        'sub_id': sub_id,
        'eeg': x,
        'audio': y,
        'w_lbs': w_lbs,
    }
    print(f"sub_id {trial['sub_id']} - story_id {trial['story_uid']}: skipped recordings: {skipped}/{len(word_events)}")
    return trial


def _extract_parral( raw, event, info, offset, n_fft, audio_embedding_length,  subs, segment_lengths, durations, word_count):    
    print('Thread:', threading.current_thread().name, 'started _extract_parral')
    word_index = {}
    generate_embeddings = SpeechEmbeddings()
    raw: Raw
    print('event',type(event))
    event= event.events() #  : pd.DataFrame    
    print('event',type(event))
    
    word_events: pd.DataFrame = event[event['kind'] == 'word']
    # raw.annotations.to_data_frame().info()
    # descs = [json.loads(desc.replace("'", "\"")) for desc in raw.annotations.description]
    # starts = [desc['start'] for desc in descs if desc['kind'] == 'word']
    sub_id = info['subject_info']['his_id']    
    story_id = float(word_events.iloc[0]['story_uid'])
    x = []
    y = []
    w_lbs = []
    skipped = 0
    for i, word_event in word_events.iterrows():    
        raw_start, word_start, duration, word_label, wav_path = word_event['start'], word_event['audio_start'], word_event['duration'], word_event['word'], word_event['sound'] 
        wav_path = wav_path.lower()
        wav_path = os.path.join('/projects/SilSpeech/Dev/SilentSpeech_Se2/listen_meg_eeg_preprocess/brainmagick/data/gwilliams2022_newdl/download/', wav_path)
        raw_start = raw_start + offset
        raw_end = raw_start + duration
        t_idxs = raw.time_as_index([raw_start, raw_end])
        data, times = raw[:, t_idxs[0]:t_idxs[1]] 
        segment_lengths.append(data.shape[-1])
        durations.append(duration)
        if data.shape[-1] <= n_fft // 4:
            skipped += 1
            continue
        #     # pad with zeros if too short for window
        #     to_pad = n_fft - data.shape[-1]
        #     data = np.pad(data, ((0, 0), (0, to_pad)), "constant")
        #     # continue        
        spectrums: Spectrum = raw.compute_psd(tmin=raw_start, tmax=raw_end, fmax=60, 
                                                n_fft=n_fft, verbose=False, n_per_seg=n_fft // 2, n_overlap=n_fft // 4)
        # spectrums.plot(picks="data", exclude="bads", amplitude=False)
        #     spectrums.plot_topomap(bands = {'Delta (0-4 Hz)': (0, 4), 'Theta (4-8 Hz)': (4, 8),
        #  'Alpha (8-12 Hz)': (8, 12), 'Beta (12-30 Hz)': (12, 30),
        #  'Gamma (30-45 Hz)': (30, 45)}, normalize=True)
        data, freqs = spectrums.get_data(return_freqs=True)
        audio_embedding = generate_embeddings.get_audio_embeddings(wav_path, word_start, duration,
                                                                    audio_embedding_length=audio_embedding_length)
        if word_label in word_index:
            word_id = word_index[word_label]
        else:
            word_id = word_count
            word_index[word_label] = word_count
            word_count += 1
        # print(f'word: {word_label}, audio features: {torch.tensor(audio_label).shape}, PSD: {torch.tensor(data).shape}')
        x.append(data)
        y.append(audio_embedding)
        w_lbs.append(word_id)
    x = torch.tensor(np.array(x)).to(torch.float32)
    y = torch.stack(y).to(torch.float32)
    w_lbs = torch.tensor(w_lbs).to(torch.int64)
    trial = {
        'story_uid': story_id,
        'sub_id': sub_id,
        'eeg': x,
        'audio': y,
        'w_lbs': w_lbs,
    }
    print(f"sub_id {trial['sub_id']} - story_id {trial['story_uid']}: skipped recordings: {skipped}/{len(word_events)}")
    return trial, word_index
def preprocess_recordings_pararrel(raws, events, infos, word_index=None, offset = 0., n_fft = 64, audio_embedding_length=30, generate_embeddings=None, **kwargs) -> Tuple[dict, dict, dict]:   
    subs = defaultdict(list)
    word_index = word_index or {}
    word_count = 0.
    logger.info('Preprocessing recordings...')
    segment_lengths = []
    durations = []
    min_audio_embedding = float('inf')
    max_audio_embedding = 0
    logger.info('Preprocessing recordings in Parallel...')
    # for raw, event, info in tqdm(zip(raws, events, infos)):
    #     trial = _extract(raw, event, info, word_index, offset, n_fft, audio_embedding_length, subs, segment_lengths, durations, word_count)


    #     subs[trial['sub_id']].append(trial)        
    #     # print(trial)
    #     for k in trial:
    #         if isinstance(trial[k], torch.Tensor):
    #             print(k, trial[k].shape)
    #         elif isinstance(trial[k], float):
    #             print(k, trial[k])
    #         else:
    #             print(k, type(trial[k]))
    # word_index.update({i: w for w, i in word_index.items()})



    results = Parallel(n_jobs=1)(delayed(_extract_parral)(raw, event, info, offset, n_fft, audio_embedding_length, subs, segment_lengths, durations, word_count) for raw, event, info in zip(raws, events, infos))


    for trial, trial_words in results:
        subs[trial['sub_id']].append(trial)
        for word in trial_words:
            if word in word_index:
                word_index[word] += trial_words[word]
            else:
                word_index[word] = trial_words[word]
    

    
    # 
    logger.info('Recordings preprocessed successfully.')
    
    return subs, word_index, {'durations': durations, 'segment_lengths': segment_lengths}

# @wrap_non_picklable_objects
def _extract_parral_v2(params, raw, offset, n_fft, audio_embedding_length,generate_embeddings):
    # generate_embeddings = SpeechEmbeddings()
    raw_start, word_start, duration, word_label, wav_path = params
    # print("raw_start",raw_start,"word_start",word_start,"duration",duration,"word_label",word_label,"wav_path",wav_path)
    wav_path = wav_path.lower()
    wav_path = os.path.join('/projects/SilSpeech/Dev/SilentSpeech_Se2/listen_meg_eeg_preprocess/brainmagick/data/gwilliams2022_newdl/download/', wav_path)
    # print('raw_start',raw_start,type(raw_start),'offset',offset,type(offset),'duration',duration,type(duration))
    is_skipped = False
    raw_start = raw_start + offset
    raw_end = raw_start + duration
    t_idxs = raw.time_as_index([raw_start, raw_end])
    data, times = raw[:, t_idxs[0]:t_idxs[1]]
    segment_len = data.shape[-1]
    if segment_len <= n_fft // 4:
        is_skipped = True
        # print('skipped')
        return is_skipped,[], [], 0, 0
    else:
        spectrums: Spectrum = raw.compute_psd(tmin=raw_start, tmax=raw_end, fmax=60, 
                                            n_fft=n_fft, verbose=False, 
                                            n_per_seg=n_fft // 2,
                                            n_overlap=n_fft // 4)

        data, freqs = spectrums.get_data(return_freqs=True)
        
        audio_embedding = generate_embeddings.get_audio_embeddings(wav_path, word_start, duration, audio_embedding_length=audio_embedding_length)
        audio_embedding = audio_embedding#.cpu()
        # print('data',data.shape,'audio_embedding',audio_embedding.shape)
        # del generate_embeddings
        return is_skipped, data, audio_embedding, segment_len, duration

def _get_pd_from_event(event):
    return event.events()

def preprocess_recordings_pararrel_v2(raws, events, infos, recording_info,  word_index=None, offset = 0., n_fft = 64, audio_embedding_length=30,save_path=None, **kwargs) -> Tuple[dict, dict, dict]:   
    subs = defaultdict(list)
    word_index = word_index or {}
    word_count = 0.
    segment_lengths = []
    durations = []
    min_audio_embedding = float('inf')
    max_audio_embedding = 0    
    event_pd = Parallel(n_jobs=32, prefer="threads")(delayed(_get_pd_from_event)(event) for event in events)
    events = event_pd
    generate_embeddings = SpeechEmbeddings()
    single_subject_recordings={}
    all_subjects = []
    for ii, recording in enumerate(recording_info):
        subject_id = recording._subject_index
        if subject_id in single_subject_recordings:
            single_subject_recordings[subject_id].append((raws[ii], events[ii], infos[ii], recording))
        else:
            single_subject_recordings[subject_id]=[(raws[ii], events[ii], infos[ii], recording)]
        # print('recording',ii,'subject',recording._subject_index)
        # print('recording',ii,'session',recording.session)
        # print('recording',ii,'story',recording.story)
        if subject_id not in all_subjects:
            all_subjects.append(subject_id)
    for ii, sub_id in enumerate(all_subjects):        
        sub_name = None
        subject_batch = single_subject_recordings[sub_id]
        for j in range(len(subject_batch)):
            raw, event, info, recording = subject_batch[j]
            
            word_events= event[event['kind'] == 'word']
            sub_info_id = info['subject_info']['his_id']
            sub_name = sub_info_id
            story_id = float(word_events.iloc[0]['story_uid'])
            x = []
            y = []
            w_lbs = []
            skipped = 0
            for i, word_event in word_events.iterrows():        
                raw_start, word_start, duration, word_label, wav_path = word_event['start'], word_event['audio_start'], word_event['duration'], word_event['word'], word_event['sound']
                raw_start = raw_start + offset
                raw_end = raw_start + duration
                t_idxs = raw.time_as_index([raw_start, raw_end])
                data, times = raw[:, t_idxs[0]:t_idxs[1]] 
                seg_len = data.shape[-1]
                if seg_len <= n_fft // 4:
                    skipped += 1
                    continue
                # index the word category according to its appearance to the word_index
                if word_label in word_index:
                    word_id = word_index[word_label]
                else:
                    word_id = word_count
                    word_index[word_label] = word_count
                    word_count += 1
                w_lbs.append(word_id)
            print(f'sub_id {sub_id} sub_name {sub_name}- {story_id} skipped recordings: {skipped}/{len(word_events)}')
            print("word_index",len(word_index))
            word_event_rows = word_events.iterrows()
            params = []
            results = []
            for i, word_event in tqdm(word_event_rows):   
                raw_start, word_start, duration, word_label, wav_path = word_event['start'], word_event['audio_start'], word_event['duration'], word_event['word'], word_event['sound']
                # params.append((float(raw_start), float(word_start), float(duration), str(word_label), str(wav_path)))
                p = (float(raw_start), float(word_start), float(duration), str(word_label), str(wav_path))
                is_skipped, data, audio_embedding, segment_len, duration = _extract_parral_v2(p, raw, offset, n_fft, audio_embedding_length, generate_embeddings)
                x.append(data)
                y.append(audio_embedding)
                segment_lengths.append(segment_len)
                durations.append(duration)
                # convert to list
                # results = Parallel(n_jobs=2)(delayed(_extract_parral_v2)(p, raw, offset, n_fft, audio_embedding_length) for p in params)
                # with ThreadPoolExecutor(max_workers=4) as executor:
                #     results = list(executor.map(_extract_parral_v2, [p for p in params], [raw for _ in range(len(params))], [offset for _ in range(len(params))], [n_fft for _ in range(len(params))], [audio_embedding_length for _ in range(len(params))]))  
                # word_event, raw, offset, n_fft, audio_embedding_length, segment_lengths, durations
                # for is_skipped, data, audio_embedding, segment_len, duration in results:
                if is_skipped:
                    # skipped += 1
                    continue                
            x = torch.tensor(np.array(x)).to(torch.float32).cpu()
            y = torch.stack(y).to(torch.float32).cpu()
            w_lbs = torch.tensor(w_lbs).to(torch.int64).cpu()
            print('x',x.shape,'y',y.shape,'w_lbs',w_lbs.shape)
            trial = {
                'story_uid': story_id,
                'sub_id': sub_info_id,
                'eeg': x,
                'audio': y,
                'w_lbs': w_lbs,
            }
            
            subs[sub_info_id].append(trial)
            print(f'sub_id {sub_id} sub_name {sub_name}- {story_id}: skipped recordings: {skipped}/{len(word_events)}')
        subj_save_path = os.path.join(base, save_path, f'{sub_name}.pt')
        print(f"finish preprocessing subject{sub_name},  Saving tensors to ",subj_save_path)
        torch.save(subs[sub_name], os.path.join(save_path, f'{sub_name}.pt'))
        # clear GPU memory
        torch.cuda.empty_cache() 

    word_index.update({i: w for w, i in word_index.items()})
    print('Recordings preprocessed successfully.')    
    return subs, word_index, {'durations': durations, 'segment_lengths': segment_lengths}


def preprocess_recordings(raws, events, infos, word_index=None, offset = 0., n_fft = 64, audio_embedding_length=30, **kwargs) -> Tuple[dict, dict, dict]:
    generate_embeddings = SpeechEmbeddings()
    subs = defaultdict(list)
    word_index = word_index or {}
    word_count = 0.
    logger.info('Preprocessing recordings...')
    segment_lengths = []
    durations = []
    min_audio_embedding = float('inf')
    max_audio_embedding = 0

    for raw, event, info in tqdm(zip(raws, events, infos)):
        raw: Raw
        event: pd.DataFrame
        event = event.events()
        word_events: pd.DataFrame = event[event['kind'] == 'word']
        # raw.annotations.to_data_frame().info()
        # descs = [json.loads(desc.replace("'", "\"")) for desc in raw.annotations.description]
        # starts = [desc['start'] for desc in descs if desc['kind'] == 'word']
        sub_id = info['subject_info']['his_id']
        
        story_id = float(word_events.iloc[0]['story_uid'])

        x = []
        y = []
        w_lbs = []
        skipped = 0
        for i, word_event in word_events.iterrows():
        
            raw_start, word_start, duration, word_label, wav_path = word_event['start'], word_event['audio_start'], word_event['duration'], word_event['word'], word_event['sound'] 

            wav_path = wav_path.lower()
            wav_path = os.path.join('/projects/SilSpeech/Dev/SilentSpeech_Se2/listen_meg_eeg_preprocess/brainmagick/data/gwilliams2022_newdl/download/', wav_path)

            raw_start = raw_start + offset
            raw_end = raw_start + duration

            t_idxs = raw.time_as_index([raw_start, raw_end])
            data, times = raw[:, t_idxs[0]:t_idxs[1]] 

            segment_lengths.append(data.shape[-1])
            durations.append(duration)
            if data.shape[-1] <= n_fft // 4:
                skipped += 1
                continue
            #     # pad with zeros if too short for window
            #     to_pad = n_fft - data.shape[-1]
            #     data = np.pad(data, ((0, 0), (0, to_pad)), "constant")
            #     # continue            
            spectrums: Spectrum = raw.compute_psd(tmin=raw_start, tmax=raw_end, fmax=60, n_jobs=32,
                                                  n_fft=n_fft, verbose=False, n_per_seg=n_fft // 2, n_overlap=n_fft // 4)
            # spectrums.plot(picks="data", exclude="bads", amplitude=False)
            #     spectrums.plot_topomap(bands = {'Delta (0-4 Hz)': (0, 4), 'Theta (4-8 Hz)': (4, 8),
            #  'Alpha (8-12 Hz)': (8, 12), 'Beta (12-30 Hz)': (12, 30),
            #  'Gamma (30-45 Hz)': (30, 45)}, normalize=True)
            data, freqs = spectrums.get_data(return_freqs=True)

            audio_embedding = generate_embeddings.get_audio_embeddings(wav_path, word_start, duration,
                                                                       audio_embedding_length=audio_embedding_length)
            audio_embedding = audio_embedding.cpu()
            audio_embedding = audio_embedding.cpu()
            
            if word_label in word_index:
                word_id = word_index[word_label]
            else:
                word_id = word_count
                word_index[word_label] = word_count
                word_count += 1
            print('word_index',len(word_index))
            # print(f'word: {word_label}, audio features: {torch.tensor(audio_label).shape}, PSD: {torch.tensor(data).shape}')
            x.append(data)
            y.append(audio_embedding)
            w_lbs.append(word_id)

        x = torch.tensor(np.array(x)).to(torch.float32).cpu()
        y = torch.stack(y).to(torch.float32).cpu()
        w_lbs = torch.tensor(w_lbs).to(torch.int64).cpu()

        trial = {
            'story_uid': story_id,
            'sub_id': sub_id,
            'eeg': x,
            'audio': y,
            'w_lbs': w_lbs,
        }
        subs[sub_id].append(trial)


        print(f'{sub_id} - {story_id}: skipped recordings: {skipped}/{len(word_events)}')
    word_index.update({i: w for w, i in word_index.items()})
    logger.info('Recordings preprocessed successfully.')
    
    return subs, word_index, {'durations': durations, 'segment_lengths': segment_lengths}



def preprocess_words(save_dir='preprocessed', **kwargs):
    save_path = os.path.join(base, save_dir)
    os.makedirs(save_path, exist_ok=True)
    time0=time.time()
    raws, events, infos, recording_info = get_raw_events(**kwargs)
    print('Finished getting raw events in:', time.time()-time0) 
    preprocess_raws_parallel(raws, **kwargs)
    word_index = {}
    
    # if os.path.exists(word_index_path):
    #     with open(word_index_path, "rb") as f:
    #         word_index = torch.load(f, weights_only=True)
    subs, word_index, additional_info = preprocess_recordings(raws, events, infos, word_index, **kwargs)
    # subs, word_index, additional_info = preprocess_recordings_pararrel_v2(raws, events, infos, recording_info, word_index, save_path=save_dir, **kwargs)    


    word_index_path = os.path.join(save_path, f'word_index.pt')
    print(f'Save preprocessed content to {word_index_path}.')
    with open(word_index_path, "wb") as f:
        torch.save(word_index, f)
    
    return subs, word_index, additional_info

def preprocess_words_test(**kwargs):
    start_time = time.time()
    # make dir 
    save_path = os.path.join(base, 'preprocessed_Nov_11')
    os.makedirs(save_path, exist_ok=True)
    subs, word_index, additional_info = preprocess_words(save_dir='preprocessed_Nov_11', **kwargs)
    # done preprocessing, now let's test if the data is saved correctly
    print("done preprocessing!")

    # save_path = os.path.join(base, 'preprocessed_Nov_11')
    for sub in subs:
        saved = torch.load(os.path.join(save_path, f'{sub}.pt'), weights_only=True)
        for trial, trial_c in zip(subs[sub], saved):
            assert len(trial['eeg']) == len(trial_c['eeg'])
    saved = torch.load(os.path.join(save_path, f'word_index.pt'), weights_only=True)
    assert len(word_index) == len(saved)
    print('Tests passed successfully.')
    print('total time:', time.time()-start_time)



    # for sub in subs:     
    #     subj_save_path = os.path.join(base, 'preprocessed_Nov_11', f'{sub}.pt')
    #     torch.save(subs,subj_save_path)   
    #     # print("Saving tensors to ", os.path.join(save_path, f'{sub}.pt'))
    #     saved = torch.load(save_path, weights_only=True)
    #     for trial, trial_c in zip(subs[sub], saved):
    #         assert len(trial['eeg']) == len(trial_c['eeg'])
    # # saved = torch.load(os.path.join(save_path, f'word_index.pt'), weights_only=True)
    # # assert len(word_index) == len(saved)
    # logger.info('Tests passed successfully.')


def run(args):
    kwargs: tp.Dict[str, tp.Any]
    kwargs = OmegaConf.to_container(args.dset, resolve=True)  # type: ignore
    selections = [args.selections[x] for x in args.dset.selections]
    kwargs["selections"] = selections
    if args.optim.loss == "clip":
        kwargs['extra_test_features'].append("WordHash")

    # return get_raw_events(**kwargs, num_workers=args.num_workers)
    # return preprocess_words_test(**kwargs, num_workers=args.num_workers)

    kwargs['offset'] = 0.15

    return preprocess_words(**kwargs, num_workers=args.num_workers, random_noise=True)

# @hydra_main(config_name="config", config_path="conf", version_base="1.1")
def main(args: tp.Any) -> float:
    print('hello there good sir.')
    override_args_(args)
    global __file__  # pylint: disable=global-statement,redefined-builtin
    # Fix bug when using multiprocessing with Hydra
    __file__ = hydra.utils.to_absolute_path(__file__)
    from bm import env 
    # from . import env  # we need this here otherwise submitit pickle does crazy stuff.
    # Updating paths in config that should stay relative to the original working dir
    # with env.temporary_from_args(args):
    #     torch.set_num_threads(1)
    #     logger.info(f"For logs, checkpoints and samples, check {os.getcwd()}.")
    #     logger.info(f"Caching intermediate data under {args.cache}.")
    #     logger.debug(args)
    return run(args)
    if '_BM_TEST_PATH' in os.environ:
        main.dora.dir = Path(os.environ['_BM_TEST_PATH'])

# CUDA_VISIBLE_DEVICES=1 python explore_dataset.py
# CUDA_VISIBLE_DEVICES=0 python explore_dataset.py
# if __name__ == "__main__":
#     with initialize(version_base="1.1", config_path="conf"):
#         cfg = compose(config_name="config.yaml", overrides=['+HYDRA_FULL_ERROR=1'])
#     configure_logging()
#     main(cfg)