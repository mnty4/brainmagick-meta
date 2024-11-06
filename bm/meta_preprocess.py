import torch.nn.functional as F
import pickle
import json
from sklearn.model_selection import train_test_split
from collections import defaultdict
from mne.time_frequency import Spectrum
from sklearn.model_selection import ShuffleSplit
import pandas as pd
from mne.io import Raw, RawArray
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
from typing import Tuple
import os
from functools import lru_cache
import functools
from tqdm import tqdm
import torch.utils
import torch.utils.data
from bm.setup_logging import configure_logging
# from . import env
# from .cache import Cache
from .dataset import _extract_recordings, _preload, assign_blocks, SegmentDataset
from .train import override_args_
from .speech_embeddings import SpeechEmbeddings
from frozendict import frozendict

from dora.log import LogProgress

logger = logging.getLogger(__name__)

base = os.path.dirname(os.path.abspath(__file__))

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
    all_recordings = _extract_recordings(
        selections, n_recordings, skip_recordings=skip_recordings,
    shuffle_recordings_seed=shuffle_recordings_seed)
    all_recordings = LogProgress(logger, all_recordings,
                                      name="Preparing cache", level=logging.DEBUG)
    all_recordings = [  # for debugging
        _preload(s, sample_rate=sample_rate, highpass=highpass) for s in all_recordings]
    
    print(all_recordings)

    raws = [recording.preprocessed(128, 0).load_data() for recording in all_recordings]
    events = [recording.events() for recording in all_recordings]
    infos = [recording.mne_info for recording in all_recordings]
    logger.info(f'Recordings loaded succesfully.')
    return raws, events, infos

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

# inplace
def preprocess_raws(raws):
    for raw in raws:
        data = raw.get_data()

        data = apply_baseline(raw, data)
        data = apply_clamp(data)
        data = normalise(data)

        raw._data = data
        

def preprocess_words(save_dir='preprocessed', **kwargs):
    save_path = os.path.join(base, save_dir)
    os.makedirs(save_path, exist_ok=True)

    raws, events, infos = get_raw_events(**kwargs)

    preprocess_raws(raws)
    
    word_index = {}
    word_index_path = os.path.join(save_path, f'word_index.pt')
    if os.path.exists(word_index_path):
        with open(word_index_path, "rb") as f:
            word_index = torch.load(f, weights_only=True)

    subs, word_index, additional_info = preprocess_recordings(raws, events, infos, word_index, **kwargs)
    logger.info(f'Saving tensors to "{save_path}"...')
    for sub in subs:
        torch.save(subs[sub], os.path.join(save_path, f'{sub}.pt'))

    with open(word_index_path, "wb") as f:
        torch.save(word_index, f)
    logger.info(f'Save successful.')
    return subs, word_index, additional_info

def split_dataset(subs: dict, seed, by_trial=False, **kwargs):
    subs_list = subs.values()
    subs_ids = subs.keys()
    task_t = np.array(subs_list)
    # if by_trial:
    #     task_t = task_t.flatten(0, 1)

    tmp_subs, tmp_sub_ids, test_subs, test_sub_ids = train_test_split(task_t, subs_ids, test_size=0.2, random_state=seed, shuffle=True)
    train_subs, train_sub_ids, valid_subs, valid_sub_ids = train_test_split(tmp_subs, tmp_sub_ids, test_size=0.125, random_state=seed, shuffle=True)
    
    # return train, valid, test
    

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

            raw_start = raw_start + offset
            end = raw_start + duration + offset

            t_idxs = raw.time_as_index([raw_start, end])
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
            
            spectrums: Spectrum = raw.compute_psd(tmin=raw_start, tmax=end, fmax=60, 
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
        subs[sub_id].append(trial)
        logger.info(f'{sub_id} - {story_id}: skipped recordings: {skipped}/{len(word_events)}')
    word_index.update({i: w for w, i in word_index.items()})
    logger.info('Recordings preprocessed successfully.')
    
    return subs, word_index, {'durations': durations, 'segment_lengths': segment_lengths}

def preprocess_words_test(**kwargs):
    subs, word_index, additional_info = preprocess_words(save_dir='preprocessed', **kwargs)
    save_path = os.path.join(base, 'preprocessed')
    for sub in subs:
        saved = torch.load(os.path.join(save_path, f'{sub}.pt'), weights_only=True)
        for trial, trial_c in zip(subs[sub], saved):
            assert len(trial['eeg']) == len(trial_c['eeg'])
    saved = torch.load(os.path.join(save_path, f'word_index.pt'), weights_only=True)
    assert len(word_index) == len(saved)
    logger.info('Tests passed successfully.')


def run(args):
    kwargs: tp.Dict[str, tp.Any]
    kwargs = OmegaConf.to_container(args.dset, resolve=True)  # type: ignore
    selections = [args.selections[x] for x in args.dset.selections]
    kwargs["selections"] = selections
    if args.optim.loss == "clip":
        kwargs['extra_test_features'].append("WordHash")

    # return get_raw_events(**kwargs, num_workers=args.num_workers)
    # return preprocess_words_test(**kwargs, num_workers=args.num_workers)
    return preprocess_words(**kwargs, num_workers=args.num_workers)

# @hydra_main(config_name="config", config_path="conf", version_base="1.1")
def main(args: tp.Any) -> float:
    print('hello there good sir.')
    override_args_(args)

    global __file__  # pylint: disable=global-statement,redefined-builtin
    # Fix bug when using multiprocessing with Hydra
    __file__ = hydra.utils.to_absolute_path(__file__)

    from . import env  # we need this here otherwise submitit pickle does crazy stuff.
    # Updating paths in config that should stay relative to the original working dir
    with env.temporary_from_args(args):
        torch.set_num_threads(1)
        logger.info(f"For logs, checkpoints and samples, check {os.getcwd()}.")
        logger.info(f"Caching intermediate data under {args.cache}.")
        logger.debug(args)
        return run(args)


    if '_BM_TEST_PATH' in os.environ:
        main.dora.dir = Path(os.environ['_BM_TEST_PATH'])

if __name__ == "__main__":
    with initialize(version_base="1.1", config_path="conf"):
        cfg = compose(config_name="config.yaml", overrides=['+HYDRA_FULL_ERROR=1'])
    configure_logging()
    main(cfg)