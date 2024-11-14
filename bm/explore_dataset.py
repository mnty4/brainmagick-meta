from sklearn.model_selection import ShuffleSplit
import pandas as pd
from mne.io import Raw
import json
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from omegaconf import OmegaConf
import logging
import typing as tp
import torch
import functools, sys, os
from functools import lru_cache

import torch.utils
import torch.utils.data
from hydra import initialize, compose
import hydra
from speech_embeddings import SpeechEmbeddings
from bm.setup_logging import configure_logging


# Redirect print statements to a file
# sys.stdout = open('output_explore_dataset.log', 'w')
# Configure logging to write to the same file
# logging.basicConfig(filename='output_explore_dataset.log', 
#                     level=logging.INFO, 
#                     format='%(asctime)s - %(levelname)s - %(message)s')
# # Example usage
# print("This is a print statement.")
# logging.info("This is an info log message.")
# logging.warning("This is a warning log message.")



"""
# from bm import env
bm.env Env(cache=None,feature_models=None,studies={'gwilliams2022': PosixPath('/projects/SilSpeech/Dev/SilentSpeech_Se2/listen_meg_eeg_preprocess/brainmagick/data/gwilliams2022_newdl'), 'schoffelen2019': PosixPath('/projects/SilSpeech/Dev/SilentSpeech_Se2/listen_meg_eeg_preprocess/brainmagick/data/schoffelen2019')})
"""
# print("bm.env",env)
# exit(0)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
print("SCRIPT_DIR",SCRIPT_DIR)
sys.path.append(os.path.dirname(SCRIPT_DIR))
# from . import env
# from .cache import Cache

from bm.dataset import _extract_recordings, _preload, assign_blocks, SegmentDataset
from bm.train import override_args_
from meta_preprocess import get_raw_events,preprocess_words_test
from meta_preprocess2 import preprocess_words_test as preprocess_words_test_2
from frozendict import frozendict
import mne
mne.set_log_level("ERROR")
# from bm.speech_embeddings import SpeechEmbeddings
# 
print("explore_dataset.py import ok")
# from dora.log import LogProgress
logger = logging.getLogger(__name__)
# from dora import hydra_main

def get_dataset(**kwargs):
    raws, events, info = get_raw_events(**kwargs)
    exit(0)
    # train, val, test = split_subjects(raws, events)
    dataset = MetaDataset(raws, events, offset=0.)
    return dataset

def get_dataloaders(train_dset, val_dset):
    train_dataloader = DataLoader(train_dset, batch_size=64, shuffle=True)
    val_dataloader = DataLoader(val_dset, batch_size=64, shuffle=True)
    return train_dataloader, val_dataloader

    # get targets

    # meg_dimension = max(recording.meg_dimension for recording in all_recordings)
    # factory_kwargs.update(sample_rate=sample_rate, highpass=highpass, meg_dimension=meg_dimension,
    #                       baseline=(None, 0) if apply_baseline else None)


    # fact = SegmentDataset.Factory(features=features, **factory_kwargs)
    # for key, value in test.items():
    #     if value is not None:
    #         factory_kwargs[key] = value
    # fact_test = SegmentDataset.Factory(features=features + extra_test_features, **factory_kwargs)

    # factories = [fact_test, fact, fact]

    # dsets_per_split: tp.List[tp.List[SegmentDataset]] = [[], [], []]
    # for i, recording in enumerate(all_recordings):
    #     events = recording.events()
    #     blocks = events[events.kind == 'block']

    #     if min_block_duration > 0 and not force_uid_assignement:
    #         if recording.study_name() not in ['schoffelen2019']:
    #             blocks = blocks.event.merge_blocks(min_block_duration_s=min_block_duration)

    #     blocks = assign_blocks(
    #         blocks, [test_ratio, valid_ratio], remove_ratio=remove_ratio, seed=split_assign_seed,
    #         min_n_blocks_per_split=min_n_blocks_per_split)
    #     for j, (fact, dsets) in enumerate(zip(factories, dsets_per_split)):
    #         split_blocks = blocks[blocks.split == j]
    #         if not split_blocks.empty:
    #             start_stops = [(b.start, b.start + b.duration) for b in split_blocks.itertuples()]
    #             dset = fact.apply(recording, blocks=start_stops)
    #             if dset is not None:
    #                 dsets.append(dset)
    #             else:
    #                 logger.warning(f'Empty blocks for split {j + 1}/{len(factories)} of '
    #                                f'recording {i + 1}/{n_recordings}.')
    #         else:
    #             logger.warning(f'No blocks found for split {j + 1}/{len(factories)} of '
    #                            f'recording {i + 1}/{n_recordings}.')
    # print(dsets_per_split)

class MetaDataset(Dataset):
    def __init__(self, raws, events, transform=None, offset = 0., **kwargs) -> None:

        self.transform = transform
        generate_embeddings = SpeechEmbeddings()
        datasets = []
        for raw, event in zip(raws, events):
            raw: Raw
            word_events: pd.DataFrame = event[event['kind'] == 'word']
            descs = [json.loads(desc.replace("'", "\"")) for desc in raw.annotations.description]
            starts = [desc['start'] for desc in descs if desc['kind'] == 'word']
            x = []
            y = []
            w_lbs = []
            
            for (i, word_event), start in zip(word_events.iterrows(), starts):

                # if i >= 100: 
                #     break
                # print(raw.annotations.description)
                # print(raw.annotations.description[2])
                # print(word_event)
            
                duration, word_label, wav_path = word_event['duration'], word_event['word'], word_event['sound'] 

                wav_path = wav_path.lower()
                
                if duration < 0.05:
                    continue

                start = start + offset
                end = start + duration + offset
                print(start, end, wav_path, word_label)
                t_idxs = raw.time_as_index([start, end])
                data, times = raw[:, t_idxs[0]:t_idxs[1]] 
                audio_label = generate_embeddings.get_audio_embeddings(wav_path, start, duration)
 
                x.append(data)
                y.append(audio_label)
                w_lbs.append(word_label)

            datasets.append(TrialDataset(x, y, w_lbs, **kwargs))
        # load datasets into self.datasets
        self.datasets = datasets
        self.len = len(self.datasets)
        print("len of datasets", self.len)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        dset = self.datasets[idx]
        if self.transform:
            dset = self.transform(dset)
        return dset

class TrialDataset(Dataset):
    def __init__(self, x, y, word_labels, n_supp=32, n_query=32, **kwargs) -> None:
        # self.x = x
        # self.y = y
        # self.word_labels = word_labels

        # self.n_supp = n_supp
        # self.n_query = n_query
        self.samples_per_batch = n_supp + n_query

        self.batches = self.create_batches(x, y, word_labels, n_supp, n_query)
        self.len = len(self.batches)
    
    def create_batches(self, x, y, word_labels, n_supp, n_query):
        n = len(x)
        batches = []
        total_batches = n // self.samples_per_batch
        for i in range(total_batches + 1):

            supp = []
            for j in range(n_supp):
                ii = i * self.samples_per_batch + j
                if ii >= n:
                    break
                supp.append({'x': x[ii], 'y': y[ii], 'word_label': word_labels[ii]})

            query = []
            for j in range(n_query):
                ii = i * self.samples_per_batch + n_supp + j
                if ii >= n:
                    break
                query.append({'x': x[ii], 'y': y[ii], 'word_label': word_labels[ii]})

            if query:
                batches.append({'supp': supp, 'query': query})

        return batches

    def create_batches2(self, x, y, word_labels, n_supp, n_query):
        # n batches in a list
        # within a batch is n_supp + n_query
        # total_batches = len(x) // self.samples_per_batch
        
        # x_np = np.array(x).reshape(total_batches, samples_per_batch, -1)

        n = len(x)
        batches = []
        batch = []
        for i in range(n):
            if i > 0 and i % self.samples_per_batch == 0:
                batches.append(batch)
                batch = []
            batch.append({'x': x[i], 'y': y[i], 'word_label': word_labels[i]})
        if batch:
            batches.append(batch)
        return batches

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.batches[idx]

    # def __getitem__(self, idx):
    #     # self.samples_per_batch = 5
    #     # idx = 3
    #     # [15]

    #     start = self.samples_per_batch * idx 
    #     end = self.samples_per_batch * idx + self.samples_per_batch
    #     self.X[0]
    #     self.Y[0]
    #     self.word_label[0]
    #     # get x[0] = brain wave clip, y[0] = audio clip
    #     # where C is channels, F is features (NOT frequency), we don't perform fourier transform.
    #     # x[0]: CxT
    #     # y[0]: FxT
    #     # y^[0]: FxT 
    #     pass

def run(args):
    kwargs: tp.Dict[str, tp.Any]
    kwargs = OmegaConf.to_container(args.dset, resolve=True)  # type: ignore
    selections = [args.selections[x] for x in args.dset.selections]
    kwargs["selections"] = selections
    if args.optim.loss == "clip":
        kwargs['extra_test_features'].append("WordHash")

    return get_dataset(**kwargs, num_workers=args.num_workers)

# @hydra_main(config_name="config", config_path="conf", version_base="1.1")
def main(args: tp.Any) -> float:
    print('hello there good sir.')
    override_args_(args)
    global __file__  # pylint: disable=global-statement,redefined-builtin
    # Fix bug when using multiprocessing with Hydra
    __file__ = hydra.utils.to_absolute_path(__file__)
    from bm import env  # we need this here otherwise submitit pickle does crazy stuff.
    # Updating paths in config that should stay relative to the original working dir
    with env.temporary_from_args(args):
        torch.set_num_threads(1)
        logger.info(f"For logs, checkpoints and samples, check {os.getcwd()}.")
        logger.info(f"Caching intermediate data under {args.cache}.")
        logger.debug(args)
        print("running run")
        return run(args)
    if '_BM_TEST_PATH' in os.environ:
        main.dora.dir = Path(os.environ['_BM_TEST_PATH'])

def explore_Gwilliams2022():
    with initialize(version_base="1.1", config_path="conf"):
        cfg = compose(config_name="config.yaml", overrides=['+HYDRA_FULL_ERROR=1'])
        # print(OmegaConf.to_yaml(cfg))  
    dset = main(cfg)    
    for i in range(10):
        print(dset[i],type(dset[i]))
        break



def run_preprocess_gwilliams(args):
    kwargs: tp.Dict[str, tp.Any]
    kwargs = OmegaConf.to_container(args.dset, resolve=True)  # type: ignore
    selections = [args.selections[x] for x in args.dset.selections]
    kwargs["selections"] = selections
    if args.optim.loss == "clip":
        kwargs['extra_test_features'].append("WordHash")

    # return get_raw_events(**kwargs, num_workers=args.num_workers)
    return preprocess_words_test(**kwargs, num_workers=args.num_workers)

def preprocess_Gwilliams2022():
    with initialize(version_base="1.1", config_path="conf"):
        cfg = compose(config_name="config.yaml", overrides=['+HYDRA_FULL_ERROR=1'])
    configure_logging()
    args = cfg
    override_args_(args)
    global __file__  # pylint: disable=global-statement,redefined-builtin
    # Fix bug when using multiprocessing with Hydra
    __file__ = hydra.utils.to_absolute_path(__file__)

    from bm import env  # we need this here otherwise submitit pickle does crazy stuff.
    # Updating paths in config that should stay relative to the original working dir
    with env.temporary_from_args(args):
        torch.set_num_threads(1)
        logger.info(f"For logs, checkpoints and samples, check {os.getcwd()}.")
        logger.info(f"Caching intermediate data under {args.cache}.")
        logger.debug(args)
        return run_preprocess_gwilliams(args)

    if '_BM_TEST_PATH' in os.environ:
        main.dora.dir = Path(os.environ['_BM_TEST_PATH'])


def run_preprocess_gwilliams_2(args):
    kwargs: tp.Dict[str, tp.Any]
    kwargs = OmegaConf.to_container(args.dset, resolve=True)  # type: ignore
    selections = [args.selections[x] for x in args.dset.selections]
    kwargs["selections"] = selections
    if args.optim.loss == "clip":
        kwargs['extra_test_features'].append("WordHash")

    # return get_raw_events(**kwargs, num_workers=args.num_workers)
    return preprocess_words_test_2(**kwargs, num_workers=args.num_workers)

def preprocess_Gwilliams2022_2():
    with initialize(version_base="1.1", config_path="conf"):
        cfg = compose(config_name="config.yaml", overrides=['+HYDRA_FULL_ERROR=1'])
    configure_logging()
    args = cfg
    override_args_(args)
    global __file__  # pylint: disable=global-statement,redefined-builtin
    # Fix bug when using multiprocessing with Hydra
    __file__ = hydra.utils.to_absolute_path(__file__)

    from bm import env  # we need this here otherwise submitit pickle does crazy stuff.
    # Updating paths in config that should stay relative to the original working dir
    with env.temporary_from_args(args):
        torch.set_num_threads(1)
        logger.info(f"For logs, checkpoints and samples, check {os.getcwd()}.")
        logger.info(f"Caching intermediate data under {args.cache}.")
        logger.debug(args)
        return run_preprocess_gwilliams_2(args)

    if '_BM_TEST_PATH' in os.environ:
        main.dora.dir = Path(os.environ['_BM_TEST_PATH'])


def explore_schoffelen():
    with initialize(version_base="1.1", config_path="conf"):
        cfg = compose(config_name="config_schoffelen.yaml", overrides=['+HYDRA_FULL_ERROR=1'])
        # print(OmegaConf.to_yaml(cfg))    
    dset = main(cfg)
    # for i in range(10):
    #     print(dset[i])
    #     break
    
if __name__ == "__main__":
    # explore_Gwilliams2022()
    # preprocess_Gwilliams2022()
    # preprocess_Gwilliams2022_2()
    explore_schoffelen()


    # Close the file at the end
    # sys.stdout.close()