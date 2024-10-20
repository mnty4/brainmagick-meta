import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from omegaconf import OmegaConf
from dora import hydra_main
import logging
import typing as tp
import torch
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
from frozendict import frozendict



from dora.log import LogProgress

logger = logging.getLogger(__name__)

# def get_datasets(selections: tp.List[tp.Dict[str, tp.Any]],
#         n_recordings: int,
#         test_ratio: float,
#         valid_ratio: float,
#         sample_rate: int,  # FIXME
#         remove_ratio: float = 0.,
#         highpass: float = 0,
#         shuffle_recordings_seed: int = -1,
#         skip_recordings: int = 0,
#         min_block_duration: float = 0.0,
#         force_uid_assignement: bool = True,
#         split_assign_seed: int = 12,
#         min_n_blocks_per_split: int = 20,
#         features: tp.Optional[tp.List[str]] = None,
#         extra_test_features: tp.Optional[tp.List[str]] = None,
#         apply_baseline: bool = True,
#         test: dict = {},
#         num_workers: int = 1,
#         **factory_kwargs: tp.Any):

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
def get_datasets(selections: tp.List[tp.Dict[str, tp.Any]],
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
        shuffle_recordings_seed: int = -1,
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

    all_recordings = _extract_recordings(
        selections, n_recordings, skip_recordings=skip_recordings,
    shuffle_recordings_seed=shuffle_recordings_seed)
    all_recordings = LogProgress(logger, all_recordings,
                                      name="Preparing cache", level=logging.DEBUG)
    all_recordings = [  # for debugging
        _preload(s, sample_rate=sample_rate, highpass=highpass) for s in all_recordings]
    
    print(all_recordings)

    raw = [recording._load_raw() for recording in all_recordings]
    events = [recording._load_events() for recording in all_recordings]

    events[0].info()

    print(raw)
    print(events)

    return raw, events

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
    def __init__(self, raw, events, n_shot, n_query) -> None:
        # load datasets into self.datasets
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        
        return self.datasets[0]

class TrialDataset(Dataset):
    def __init__(self, x, y, word_labels, n_supp, n_query) -> None:
        self.x = x
        self.y = y
        self.word_labels = word_labels
        self.n_supp = n_supp
        self.n_query = n_query
        self.samples_per_batch = n_supp + n_query

        self.batches = self.create_batches(x, y, word_labels)
        self.len = len(self.batches)

    def create_batches(self, x, y, word_labels):
        total_batches = len(self.x) // self.n_supp + self.n_query
        samples_per_batch = self.n_supp + self.n_query
        x_np = np.array(x).reshape(total_batches, samples_per_batch, -1)

        # n = len(x)
        # for i in range(n):
        #      = x[i], y[i], word_labels[i]


        pass
    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        # self.samples_per_batch = 5
        # idx = 3
        # [15]

        start = self.samples_per_batch * idx 
        end = self.samples_per_batch * idx + self.samples_per_batch
        self.X[0]
        self.Y[0]
        self.word_label[0]
        # get x[0] = brain wave clip, y[0] = audio clip
        # where C is channels, F is features (NOT frequency), we don't perform fourier transform.
        # x[0]: CxT
        # y[0]: FxT
        # y^[0]: FxT 
        pass


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

    return get_datasets(**kwargs, num_workers=args.num_workers)

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
    main()
    