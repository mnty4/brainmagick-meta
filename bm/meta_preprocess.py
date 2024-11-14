import pickle
import json
from sklearn.model_selection import train_test_split
from collections import defaultdict
from mne.time_frequency import Spectrum
from sklearn.model_selection import ShuffleSplit
import pandas as pd
from mne.io import Raw
import numpy as np
from pathlib import Path
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
from speech_embeddings import SpeechEmbeddings
from frozendict import frozendict
from _env import Env as env  # we need this here otherwise submitit pickle does crazy stuff.
from dora.log import LogProgress
import sys
from joblib import Parallel, delayed
import threading
# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# print("SCRIPT_DIR",SCRIPT_DIR)
# sys.path.append(os.path.dirname(SCRIPT_DIR))

logger = logging.getLogger(__name__)
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

@list_to_tuple
@deep_freeze_args
@lru_cache(typed=True)
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
    print('[meta_preprocess.get_raw_events] selections',selections)
    # (frozendict.frozendict({'study': 'gwilliams2022'}),)

    # get from running gwilliams study
    logger.info(f'Loading recordings...')
    all_recordings = _extract_recordings(
        selections, n_recordings, skip_recordings=skip_recordings,
    shuffle_recordings_seed=shuffle_recordings_seed)
    # print('all_recordings',all_recordings,selections)
    """
        all_recordings 
        [
        Gwilliams2022Recording('01_session0_story0'), 
        Gwilliams2022Recording('01_session0_story1'),
        Gwilliams2022Recording('01_session0_story2'), 
        Gwilliams2022Recording('01_session0_story3'), 
        Gwilliams2022Recording('01_session1_story0'), 
        Gwilliams2022Recording('01_session1_story1'), 
        Gwilliams2022Recording('01_session1_story2'), 
        Gwilliams2022Recording('01_session1_story3'), 
        Gwilliams2022Recording('02_session0_story0'), 
        Gwilliams2022Recording('02_session0_story1'), 
        ...        
    """
    all_recordings = LogProgress(
            logger, 
            all_recordings,        
            name="Preparing cache", 
            level=logging.DEBUG
            )
    # pre_load_Recordings = []    
    # for s in all_recordings:
    #     print('preloading',s,s.study_name(),'story',s.story, 'session',s.session,'_subject_index',s._subject_index,'_recording_index',s._recording_index)

    #     # denbugging
    #     if not s._subject_index>7 :
    #         break
    #     # if not s._subject_index==14 and not int(s.session) == 1:
    #     #     continue
    #     preloaded_data = _preload(s, sample_rate=sample_rate, highpass=highpass)
    #     print('preloaded finished')
    #     print('_cache_folder',s._cache_folder)
    #     pre_load_Recordings.append(preloaded_data)        
    # # all_recordings = [  # for debugging
    # #     _preload(s, sample_rate=sample_rate, highpass=highpass) for s in all_recordings]
    # all_recordings = pre_load_Recordings


    preloaded_data_list = Parallel(n_jobs=32)(delayed(_preload)(s, sample_rate=sample_rate, highpass=highpass) for s in all_recordings)
    all_recordings = preloaded_data_list

    # para


    # print("all_recordings",all_recordings)
    raws = [recording.preprocessed(120, 0) for recording in all_recordings]
    events = [recording.events() for recording in all_recordings]
    infos = [recording.mne_info for recording in all_recordings]
    print('len(raws)',len(raws),'len(events)',len(events),'len(infos)',len(infos))
    
    print(f'Recordings loaded succesfully.')
    return raws, events, infos

def preprocess_words(save_dir='preprocessed', **kwargs):
    save_path = os.path.join(base, save_dir)
    os.makedirs(save_path, exist_ok=True)

    raws, events, infos = get_raw_events(**kwargs)

    word_index = {}
    word_index_path = os.path.join(save_path, f'word_index.pt')
    if os.path.exists(word_index_path):
        with open(word_index_path, "rb") as f:
            word_index = torch.load(f, weights_only=True)
    subs, word_index = preprocess_recordings_pararrel(raws, events, infos, word_index, **kwargs)
    # subs, word_index = preprocess_recordings(raws, events, infos, word_index, **kwargs)
    logger.info(f'Saving tensors to "{save_path}"...')
    for sub in subs:
        torch.save(subs[sub], os.path.join(save_path, f'{sub}.pt'))

    with open(word_index_path, "wb") as f:
        torch.save(word_index, f)
    logger.info(f'Save successful.')
    return subs, word_index

def split_dataset(subs: dict, seed, by_trial=False, **kwargs):
    subs_list = subs.values()
    subs_ids = subs.keys()
    task_t = np.array(subs_list)
    # if by_trial:
    #     task_t = task_t.flatten(0, 1)

    tmp_subs, tmp_sub_ids, test_subs, test_sub_ids = train_test_split(task_t, subs_ids, test_size=0.2, random_state=seed, shuffle=True)
    train_subs, train_sub_ids, valid_subs, valid_sub_ids = train_test_split(tmp_subs, tmp_sub_ids, test_size=0.125, random_state=seed, shuffle=True)
    
    # return train, valid, test
    
def _extract(raw, event, info, offset = 0., n_fft = 60, **kwargs):
    thread_id = threading.get_ident()
    word_index = {}
    subs = defaultdict(list)
    word_count = 0
    generate_embeddings = SpeechEmbeddings() 
    print('Thread',thread_id,'_extract from raw',type(raw),'event',type(event),'info',type(info),info)

    # declare variables
    raw: Raw
    event: pd.DataFrame    
    word_events: pd.DataFrame = event[event['kind'] == 'word']
    # raw.annotations.to_data_frame().info()
    # descs = [json.loads(desc.replace("'", "\"")) for desc in raw.annotations.description]
    # starts = [desc['start'] for desc in descs if desc['kind'] == 'word']
    sub_id = info['subject_info']['his_id']

    x = []
    y = []
    w_lbs = []
    
    # for (i, word_event), start in zip(word_events.iterrows(), starts):
    for i, word_event in word_events.iterrows():
    
        raw_start, word_start, duration, word_label, wav_path = word_event['start'], word_event['audio_start'], word_event['duration'], word_event['word'], word_event['sound'] 

        wav_path = wav_path.lower()
        # add the root folder:
        # /projects/SilSpeech/Dev/SilentSpeech_Se2/listen_meg_eeg_preprocess/brainmagick/data/gwilliams2022_newdl/download/stimuli
        wav_path = os.path.join('/projects/SilSpeech/Dev/SilentSpeech_Se2/listen_meg_eeg_preprocess/brainmagick/data/gwilliams2022_newdl/download/', wav_path)
        # check if this path exists
        # print('wav_path',wav_path,'exists',os.path.exists(wav_path))
        # stimuli/audio/lw1_0.wav
        # print('wav_path',wav_path)

        raw_start = raw_start + offset
        end = raw_start + duration + offset

        t_idxs = raw.time_as_index([raw_start, end])
        data, times = raw[:, t_idxs[0]:t_idxs[1]] 
        if data.shape[-1] < n_fft:
            continue

        spectrums: Spectrum = raw.compute_psd(tmin=raw_start, tmax=end, fmax=60, n_fft=n_fft, verbose=False)
        # spectrums.plot(picks="data", exclude="bads", amplitude=False)
        data, freqs = spectrums.get_data(return_freqs=True)
        # assert len(freqs) == 8
        # print('data',type(data),'freqs',type(freqs))            
        audio_embedding = generate_embeddings.get_audio_embeddings(wav_path, word_start, duration)
        if word_label in word_index:
            word_id = word_index[word_label]
        else:
            word_id = word_count
            word_index[word_label] = word_count
            word_count += 1
        # print('Thread',thread_id, f'Extracted word: {word_label}, PSD: {torch.tensor(data).shape}, audio_embedding: {torch.tensor(audio_embedding).shape}')
        x.append(data)
        y.append(audio_embedding)
        w_lbs.append(word_id)
    x = torch.tensor(np.array(x))
    w_lbs = torch.tensor(w_lbs)
    subs[sub_id].append((x, y, w_lbs))
    return subs, word_index
        
def preprocess_recordings_pararrel(raws, events, infos, word_index=None, offset = 0., n_fft = 60, **kwargs) -> dict:
    # generate_embeddings = SpeechEmbeddings() # generator . 
    # Global variables
    subs = defaultdict(list)
    word_index = word_index or {}
    word_count = 0.
    print('[preprocess_recordings_pararrel],len(raws)',len(raws),'len(events)',len(events),'len(infos)',len(infos))
    # [preprocess_recordings_pararrel],len(raws) 196 len(events) 196 len(infos) 196
    # for debuging, let only do 3 recordings
    # raws = raws[:3]
    # events = events[:3]
    # infos = infos[:3]
    logger.info('Preprocessing recordings in Parallel...')
    results = Parallel(n_jobs=16)(delayed(_extract)(raw, event, info, offset=offset, n_fft=n_fft, **kwargs) for raw, event, info in tqdm(zip(raws, events, infos)))
    for subs, word_index in results:
        # merge results by adding to global variables  
        for sub_id in subs:
            subs[sub_id].extend(subs[sub_id])
        # merge word_index by adding 
        for word in word_index:
            if word in word_index:
                word_index[word] += word_index[word]
            else:
                word_index[word] = word_index[word]
    print('Recordings preprocessed successfully.')
    print('word_index',word_index)

    return subs, word_index

def preprocess_recordings(raws, events, infos, word_index=None, offset = 0., n_fft = 60, **kwargs) -> dict:

    generate_embeddings = SpeechEmbeddings() # generator . 
    subs = defaultdict(list)
    word_index = word_index or {}
    word_count = 0.
    logger.info('Preprocessing recordings...')
    for raw, event, info in tqdm(zip(raws, events, infos)):
        raw: Raw
        event: pd.DataFrame
        
        word_events: pd.DataFrame = event[event['kind'] == 'word']
        # raw.annotations.to_data_frame().info()
        # descs = [json.loads(desc.replace("'", "\"")) for desc in raw.annotations.description]
        # starts = [desc['start'] for desc in descs if desc['kind'] == 'word']
        sub_id = info['subject_info']['his_id']

        x = []
        y = []
        w_lbs = []
        
        # for (i, word_event), start in zip(word_events.iterrows(), starts):
        for i, word_event in word_events.iterrows():
        
            raw_start, word_start, duration, word_label, wav_path = word_event['start'], word_event['audio_start'], word_event['duration'], word_event['word'], word_event['sound'] 

            wav_path = wav_path.lower()
            # add the root folder:
            # /projects/SilSpeech/Dev/SilentSpeech_Se2/listen_meg_eeg_preprocess/brainmagick/data/gwilliams2022_newdl/download/stimuli
            wav_path = os.path.join('/projects/SilSpeech/Dev/SilentSpeech_Se2/listen_meg_eeg_preprocess/brainmagick/data/gwilliams2022_newdl/download/', wav_path)
            # check if this path exists
            # print('wav_path',wav_path,'exists',os.path.exists(wav_path))
            # stimuli/audio/lw1_0.wav
            # print('wav_path',wav_path)

            raw_start = raw_start + offset
            end = raw_start + duration + offset

            t_idxs = raw.time_as_index([raw_start, end])
            data, times = raw[:, t_idxs[0]:t_idxs[1]] 
            if data.shape[-1] < n_fft:
                continue

            spectrums: Spectrum = raw.compute_psd(tmin=raw_start, tmax=end, fmax=60, n_fft=n_fft, verbose=False)
            # spectrums.plot(picks="data", exclude="bads", amplitude=False)
            data, freqs = spectrums.get_data(return_freqs=True)
            # assert len(freqs) == 8
            # print('data',type(data),'freqs',type(freqs))            
            audio_embedding = generate_embeddings.get_audio_embeddings(wav_path, word_start, duration)
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
        x = torch.tensor(np.array(x))
        w_lbs = torch.tensor(w_lbs)
        subs[sub_id].append((x, y, w_lbs))
    logger.info('Recordings preprocessed successfully.')
    print('word_index',word_index)

    return subs, word_index

def preprocess_words_test(**kwargs):
    subs, word_index = preprocess_words(save_dir='preprocessed', **kwargs)
    save_path = os.path.join(base, 'preprocessed')
    # print("[preprocess_words_test] save_path",save_path)
    
    for sub in subs:
        saved = torch.load(os.path.join(save_path, f'{sub}.pt'), weights_only=True)
        assert len(subs[sub]) == len(saved)
    saved = torch.load(os.path.join(save_path, f'word_index.pt'), weights_only=True)
    assert len(word_index) == len(saved)

def run(args):
    kwargs: tp.Dict[str, tp.Any]
    kwargs = OmegaConf.to_container(args.dset, resolve=True)  # type: ignore
    selections = [args.selections[x] for x in args.dset.selections]
    kwargs["selections"] = selections
    if args.optim.loss == "clip":
        kwargs['extra_test_features'].append("WordHash")

    # return get_raw_events(**kwargs, num_workers=args.num_workers)
    return preprocess_words_test(**kwargs, num_workers=args.num_workers)

# @hydra_main(config_name="config", config_path="conf", version_base="1.1")
def main(args: tp.Any) -> float:
    print('hello there good sir.')
    override_args_(args)

    global __file__  # pylint: disable=global-statement,redefined-builtin
    # Fix bug when using multiprocessing with Hydra
    __file__ = hydra.utils.to_absolute_path(__file__)

    
    # Updating paths in config that should stay relative to the original working dir
    # with env.temporary_from_args(args):
    #     torch.set_num_threads(1)
    #     logger.info(f"For logs, checkpoints and samples, check {os.getcwd()}.")
    #     logger.info(f"Caching intermediate data under {args.cache}.")
    #     logger.debug(args)
    return run(args)


    if '_BM_TEST_PATH' in os.environ:
        main.dora.dir = Path(os.environ['_BM_TEST_PATH'])

if __name__ == "__main__":
    with initialize(version_base="1.1", config_path="conf"):
        cfg = compose(config_name="config.yaml", overrides=['+HYDRA_FULL_ERROR=1'])
    configure_logging()
    main(cfg)