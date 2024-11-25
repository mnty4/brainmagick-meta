from collections import defaultdict
from torch.utils.data import Dataset
import numpy as np
import torch
import os
import glob
from typing import Tuple, List, Literal
import regex as re
base = os.path.dirname(os.path.abspath(__file__))

def parse_recording_path(path: str):
    base_name = os.path.basename(path)
    try:
        sub, session, story = base_name.replace('.pt', '').split('_')
    except:
        raise ValueError(f'Invalid preprocessed recording format, expecting "sub-{{sub_id}}_{{session_id}}_{{story_Id}}.pt" but got {base_name}')
    return sub, session, story
    
def split_recordings(paths, split_strategy: Literal['sub', 'session', 'recording']='session', **kwargs) -> Tuple[List, List, List]:
    
    if split_strategy == 'sub':
        train, val, test = split_by_sub(paths, **kwargs)
    elif split_strategy == 'recording':
        train, val, test = split_by_recording(paths, **kwargs)
    elif split_strategy == 'session':
        train, val, test = split_by_session(paths, **kwargs)
    else:
        raise ValueError('Invalid split_strategy defined.')
    
    return train, val, test

def split_by_sub(paths: List[str], **kwargs):

    subs = defaultdict(list)
    for path in paths:
        sub, session, story = parse_recording_path(path)
        subs[sub].append(path)

    sub_count = len(subs)
    train_ids, val_ids, test_ids = get_train_val_test_ids(sub_count, **kwargs)
    train, val, test = [], [], []
    sub_names = list(subs.keys())
    for train_id in train_ids:
        train.extend(subs[sub_names[train_id]])
    for val_id in val_ids:
        val.extend(subs[sub_names[val_id]])
    for test_id in test_ids:
        test.extend(subs[sub_names[test_id]])

    return train, val, test

def split_by_recording(paths: List[str], **kwargs):
    recordings = paths
    recording_count = len(recordings)

    train_ids, val_ids, test_ids = get_train_val_test_ids(recording_count, **kwargs)

    train = [paths[i] for i in train_ids]
    val = [paths[i] for i in val_ids]
    test = [paths[i] for i in test_ids]

    return train, val, test

def split_by_session(paths: List[str], **kwargs):

    subs = defaultdict(lambda: defaultdict(list))
    for path in paths:
        sub, session, story = parse_recording_path(path)
        subs[sub][session].append(path)

    # flatten sessions
    sessions = []
    for sub in subs.values():
        for session in sub.values():
            sessions.append(session)
    
    session_count = len(sessions)

    train_ids, val_ids, test_ids = get_train_val_test_ids(session_count, **kwargs)
    train, val, test = [], [], []

    for train_id in train_ids:
        train.extend(sessions[train_id])
    for val_id in val_ids:
        val.extend(sessions[val_id])
    for test_id in test_ids:
        test.extend(sessions[test_id])

    return train, val, test

def get_train_val_test_ids(count: int, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, **kwargs) -> Tuple[List[int], List[int], List[int]]:
    assert count > 2

    train = max(int(count * train_ratio), 1)
    val = max(int(count * val_ratio), 1)
    test = max(int(count * test_ratio), 1)

    remainder = count - (val + train + test)

    train += remainder

    all_indices = np.arange(count)
    np.random.shuffle(all_indices)

    train_ids = all_indices[:train].tolist()
    val_ids = all_indices[train:train + val].tolist()
    test_ids = all_indices[train + val:].tolist()

    return (train_ids, val_ids, test_ids)

def get_recordings_by_strategy(is_train=True, preprocessed_dir='preprocessed', **kwargs) -> Tuple[dict, dict]:
    save_path = os.path.join(base, preprocessed_dir)
    tensor_paths = sorted(glob.glob(os.path.join(save_path, '*.pt')))
    if not tensor_paths:
        raise ValueError(f'No recordings in {save_path}')
    word_index = {}
    remove_i = None
    for i, path in enumerate(tensor_paths):
        name = os.path.basename(path)
        if name == 'word_index.pt':
            word_index = torch.load(path, weights_only=True)
            remove_i = i
    if remove_i is None:
        raise ValueError('Couldn\'t find word_index.')
    tensor_paths.pop(remove_i)
    
    splits = split_recordings(tensor_paths, **kwargs)

    splits = load_preprocessed_by_mode(splits, is_train)

    return splits, word_index

def load_preprocessed_by_mode(splits: Tuple[List[str], List[str], List[str]], is_train=True):
    train, val, test = splits
    load = lambda path: torch.load(path, weights_only=True)
    if is_train:
        train = list(map(load, train))
        val = list(map(load, val))
        test = []
    else:
        test = list(map(load, test))
        train, val = [], []
    
    return train, val, test

def load_preprocessed(is_train=True, preprocessed_dir='preprocessed', **kwargs) -> Tuple[dict, dict]:
    save_path = os.path.join(base, preprocessed_dir)
    tensor_paths = sorted(glob.glob(os.path.join(save_path, '*.pt')))
    sub_paths = []
    word_index = {}
    for path in tensor_paths:
        name = os.path.basename(path)
        if re.match('sub-\d+\.pt', name):
            sub_paths.append(path)
        if name == 'word_index.pt':
            word_index = torch.load(path, weights_only=True)

    sub_count = len(sub_paths)

    train_subs = {}
    val_subs = {}
    test_subs = {}

    # reuse the same sub if only 1 sub (this is only for development)
    if sub_count == 1:
        sub_path = sub_paths[0]
        name = os.path.basename(sub_path)
        sub = torch.load(sub_path, weights_only=True)
        train_subs[name], val_subs[name], test_subs[name] = sub, sub, sub
    
    # use 1 sub for train and reuse a separate sub for val and test (this is only for development)
    if sub_count == 2:
        train_path = sub_paths[0]
        val_test_path = sub_paths[1]
        train_name = os.path.basename(train_path)
        val_test_name = os.path.basename(val_test_path)
        train_subs[train_name] = torch.load(train_path, weights_only=True)
        val_subs[val_test_name] = torch.load(val_test_path, weights_only=True)
        test_subs[val_test_name] = val_subs[val_test_name]
    
    if sub_count < 3:
        return (train_subs, val_subs, test_subs), word_index

    train_ids, val_ids, test_ids = split_subs(sub_count)

    if is_train:
        for idx in train_ids:
            path = sub_paths[idx]
            name = os.path.basename(path)
            train_subs[name] = torch.load(path, weights_only=True)
        
        for idx in val_ids:
            path = sub_paths[idx]
            name = os.path.basename(path)
            val_subs[name] = torch.load(path, weights_only=True)
    else:
        for idx in test_ids:
            path = sub_paths[idx]
            name = os.path.basename(path)
            test_subs[name] = torch.load(path, weights_only=False)

    return (train_subs, val_subs, test_subs), word_index

def split_subs(sub_count: int, **kwargs) -> Tuple[List[int], List[int], List[int]]:
    assert sub_count > 2

    val = max((sub_count * 1) // 10, 1)
    train = max((sub_count * 7) // 10, 1)
    test = max((sub_count * 2) // 10, 1)

    remainder = sub_count - (val + train + test)

    train += remainder

    all_indices = np.arange(sub_count)
    np.random.shuffle(all_indices)

    train_ids = all_indices[:train].tolist()
    val_ids = all_indices[train:train + val].tolist()
    test_ids = all_indices[train + val:].tolist()

    return (train_ids, val_ids, test_ids)


class Trials_Dataset(Dataset):
    def __init__(self, trials, mini_batches_per_trial=1, samples_per_mini_batch=64, transform=None):
        self.transform = transform
        mini_batches = []
        samples_per_mini_batches = samples_per_mini_batch * mini_batches_per_trial
        for trial in trials:
            i = 0
            while i < len(trial['eeg']) - samples_per_mini_batches + 1:
                sub_batches = []
                for j in range(mini_batches_per_trial):
                    batch = {}
                    for key in trial:
                        if isinstance(trial[key], (torch.Tensor, list)):
                            start_idx = i + j * samples_per_mini_batch
                            end_idx = start_idx + samples_per_mini_batch
                            batch[key] = trial[key][start_idx:end_idx]
                        else:
                            batch[key] = trial[key]
                    sub_batches.append(batch)
                mini_batches.append(sub_batches)
                i += samples_per_mini_batches

        self.mini_batches_per_trial = mini_batches_per_trial
        self.samples_per_mini_batch = samples_per_mini_batch
        self.samples_per_mini_batches = samples_per_mini_batches
        self.mini_batches = mini_batches

    def __len__(self):
        return len(self.mini_batches)

    def __getitem__(self, idx):
        batch = self.mini_batches[idx]

        if self.transform:
            batch = self.transform(batch)

        return batch



def flatten_subs(subs):
    trials = []
    for sub in subs:
        trials.extend(trial for trial in subs[sub])
    return trials

def shuffle_samples(trials):
    for i, trial in enumerate(trials):
        ids = torch.randperm(trial['eeg'].shape[0])
        trials[i]['eeg'] = trials[i]['eeg'][ids]
        trials[i]['audio'] = trials[i]['audio'][ids]
        trials[i]['w_lbs'] = trials[i]['w_lbs'][ids]

def get_dataset(recordings: list, mini_batches_per_trial=1, samples_per_mini_batch=64, **kwargs):

    shuffle_samples(recordings)

    dset = Trials_Dataset(recordings, mini_batches_per_trial=mini_batches_per_trial, samples_per_mini_batch=samples_per_mini_batch)

    return dset

def get_datasets(is_train=True, train_kwargs={}, val_kwargs={}, **kwargs):
    torch.cuda.empty_cache()
    (train, val, test), word_index = get_recordings_by_strategy(is_train=is_train, **kwargs)

    if is_train:
        train_dset = get_dataset(train, **kwargs, **train_kwargs)
        val_dset = get_dataset(val, **kwargs, **val_kwargs)
        return train_dset, val_dset, word_index
    else:
        test_dset = get_dataset(test, **kwargs)
        return test_dset, word_index
    
def test_get_datasets():
    train_dset, val_dset, word_index = get_datasets(is_train=True, mini_batches_per_trial=2, samples_per_mini_batch=64, split_strategy='session')
    print(f'train len: {len(train_dset)}, val len: {len(val_dset)}, word_index len: {len(word_index)}')

if __name__ == '__main__':
    test_get_datasets()
