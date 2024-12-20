from collections import defaultdict
from sklearn.model_selection import ShuffleSplit
import pandas as pd
from mne.io import Raw
from mne.time_frequency import Spectrum
import json
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

import torch.utils
import torch.utils.data
# from . import env
# from .cache import Cache
from .dataset import _extract_recordings, _preload, assign_blocks, SegmentDataset
from .train import override_args_
from .speech_embeddings import SpeechEmbeddings
from frozendict import frozendict
from .meta_preprocess import get_raw_events

from dora.log import LogProgress

logger = logging.getLogger(__name__)

def get_dataset(**kwargs):

    raws, events, info = get_raw_events(**kwargs)

    # train, val, test = split_subjects(raws, events)

    dset = MetaDataset(raws, events, offset=0.)

    return dset


# def split(raws, events, train = 0.7, val = 0.2, test = 0.1):
#     assert len(raws) == len(events)

#     n = len(raws)

#     tr = int(train * n)
#     va = int(val * n)
#     te = int(test * n)

#     # if samples are missed, add them to test
#     te += n - (tr + va + te)

#     train_split, val_split, test_split = [], [], []

#     train_split = {'raws': raws[:]}

#     for i in range(tr):
#         train_split.append(raws[i])

    

    
        


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

def split_recordings(raws, events, infos, offset = 0., n_fft = 120):

    generate_embeddings = SpeechEmbeddings()
    subs = defaultdict(list)
    for raw, event, info in zip(raws, events, infos):
        raw: Raw
        event: pd.DataFrame
        
        word_events: pd.DataFrame = event[event['kind'] == 'word']
        raw.annotations.to_data_frame().info()
        descs = [json.loads(desc.replace("'", "\"")) for desc in raw.annotations.description]
        starts = [desc['start'] for desc in descs if desc['kind'] == 'word']
        sub_id = info['subject_info']['his_id']

        x = []
        y = []
        w_lbs = []
        
        for (i, word_event), start in zip(word_events.iterrows(), starts):
        
            duration, word_label, wav_path = word_event['duration'], word_event['word'], word_event['sound'] 

            wav_path = wav_path.lower()

            start = start + offset
            end = start + duration + offset
            
            # print(start, end, wav_path, word_label)
            t_idxs = raw.time_as_index([start, end])
            data, times = raw[:, t_idxs[0]:t_idxs[1]] 
            if data.shape[-1] < n_fft:
                continue

            spectrums: Spectrum = raw.compute_psd(tmin=start, tmax=end, fmax=60, n_fft=n_fft)
            # spectrums.plot(picks="data", exclude="bads", amplitude=False)
            data, freqs = spectrums.get_data(return_freqs=True)
            assert len(freqs) == 8
            
            # print('should have 8 freq bins: ', len(freqs))
            # preprocessed_data = to_spectrum(data)
            # bands = julius.split_bands(torch.Tensor(data), n_bands=10, sample_rate=120)

            audio_label = generate_embeddings.get_audio_embeddings(wav_path, start, duration)

            # print(f'word: {word_label}, audio features: {torch.tensor(audio_label).shape}, PSD: {torch.tensor(data).shape}')
            x.append(data)
            y.append(audio_label)
            w_lbs.append(word_label)
        # X.append(x)
        # Y.append(y)
        # word_labels.append(w_lbs)

        subs[sub_id].append((x, y, w_lbs))
    return subs

class TrialsDataset(Dataset):
    def __init__(self, trials, transform=None, **kwargs) -> None:
        self.transform = transform
        self.trials = trials

    def __len__(self):
        return len(self.trials)

    def __getitem__(self, idx):
        trial = self.trials[idx]
        if self.transform:
            sub = self.transform(sub)
        return trial
class SubDataset(Dataset):
    def __init__(self, subs, sub_ids, transform=None, **kwargs) -> None:
        self.transform = transform
        self.subs = subs
        self.sub_ids = sub_ids

    def __len__(self):
        return len(self.subs)

    def __getitem__(self, idx):
        sub, sub_id = self.subs[idx], self.sub_ids[idx]
        if self.transform:
            sub = self.transform(sub)
        return sub, sub_id

class MetaDataset_v2(Dataset):
    def __init__(self, raws, events, infos, transform=None, offset = 0., **kwargs) -> None:

        self.transform = transform
        generate_embeddings = SpeechEmbeddings()
        subs = defaultdict(list)
        for raw, event, info in zip(raws, events, infos):
            raw: Raw
            event: pd.DataFrame

            word_events: pd.DataFrame = event[event['kind'] == 'word']
            descs = [json.loads(desc.replace("'", "\"")) for desc in raw.annotations.description]
            starts = [desc['start'] for desc in descs if desc['kind'] == 'word']
            sub_id = info['subject_info']['his_id']

            x = []
            y = []
            w_lbs = []
            
            for (i, word_event), start in zip(word_events.iterrows(), starts):
            
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

            subs[sub_id].append((x, y, w_lbs))

        self.subs = subs
        self.len = len(self.subs)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        dset = self.subs[idx]
        if self.transform:
            dset = self.transform(dset)
        return dset


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
                batches.append((supp, query))

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


# dataset which takes n_shot, n_way
# class CustomDataset(Dataset):
#     def __init__(self, db, mode, k_shot, k_query) -> None:
#         super().__init__()
#         # self.data = data
#         self.db = db
#         self.mode = mode
#         self.shape = (len(db.subj[mode]), db.n_way, db.num_trials) + db.eeg_shape
#         self.n_way = self.shape[1]
#         self.k_shot = k_shot
#         self.k_query = k_query
#         self.out_shape = (self.n_way * self.k_shot,) + self.shape[-2:]
#         self.out_shape_query = (self.n_way * self.k_query,) + self.shape[-2:]
#         self.shuffle_idx = np.zeros(self.shape[:3], dtype=int)
#         for p in range(self.shape[0]):
#             for q in range(self.shape[1]):
#                 idx_range = np.arange(self.shape[2])
#                 np.random.shuffle(idx_range)
#                 self.shuffle_idx[p, q, ...] = idx_range

#     def __len__(self):
#         return self.shape[0] * (self.shape[2] // (self.k_shot + self.k_query))

#     def __getitem__(self, idx):
#         idx2 = (self.k_shot + self.k_query) * (idx // self.shape[0])
#         idx0 = idx % self.shape[0]

#         support_x = np.zeros(self.out_shape)
#         support_y = np.zeros(self.out_shape[:1], dtype=int)
#         query_x = np.zeros(self.out_shape_query)
#         query_y = np.zeros(self.out_shape_query[:1], dtype=int)

#         for j in range(self.n_way):
#             # support_x[(j*self.k_shot):((j+1)*self.k_shot), ...] = self.data[idx0][j][self.shuffle_idx[idx0, j, idx2:idx2+self.k_shot]]
#             for v in range(self.k_shot):
#                 support_x[(j*self.k_shot) + v, ...] = self.db.get_data(self.mode, self.db.subj[self.mode][idx0], j, self.shuffle_idx[idx0, j, idx2+v])
#             support_y[(j*self.k_shot):((j+1)*self.k_shot)] = j

#             # query_x[(j*self.k_query):((j+1)*self.k_query), ...] = self.data[idx0][j][self.shuffle_idx[idx0, j, idx2+self.k_shot:idx2+self.k_shot+self.k_query]]
#             for v in range(self.k_query):
#                 query_x[(j*self.k_query) + v, ...] = self.db.get_data(self.mode, self.db.subj[self.mode][idx0], j, self.shuffle_idx[idx0, j, idx2+self.k_shot+v])
#             query_y[(j*self.k_query):((j+1)*self.k_query)] = j

#         return support_x, support_y, query_x, query_y



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
    dset = main(cfg)
    