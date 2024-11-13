from collections import defaultdict
from torch.utils.data import Dataset
import numpy as np
import torch
import os
import glob
from typing import Tuple, List
import regex as re
base = os.path.dirname(os.path.abspath(__file__))

def load_preprocessed_trials(is_train=True, preprocessed_dir='preprocessed', strategy='session', **kwargs) -> Tuple[dict, dict]:
    save_path = os.path.join(base, preprocessed_dir)
    tensor_paths = sorted(glob.glob(os.path.join(save_path, '*.pt')))
    # trial_paths = []
    subs = defaultdict(lambda: ({}, {}))
    word_index = {}
    for path in tensor_paths:
        name = os.path.basename(path)
        if re.match('sub-\d+_\d+_\d+\.pt', name):
            sub, sess, story = name.split('_')
            subs[sub][int(sess)][story] = path
            # trial_paths.append(path)
        if name == 'word_index.pt':
            word_index = torch.load(path, weights_only=True)
    
    if strategy == 'session':
        sessions = flatten_subs(subs)


    # elif strategy == 'sub':


def split_by_sub(subs, is_train=True):
    sub_count = len(subs)

    train_subs = {}
    val_subs = {}
    test_subs = {}    

    train_ids, val_ids, test_ids = split_subs(sub_count)

    for sub in subs:
        for session in range(2):
            for trial_path in subs[sub][session]:
                name = os.path.basename(path)
                train_subs[name] = torch.load(path, weights_only=True)
    
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
        return train_subs, val_subs, test_subs

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

    return (train_subs, val_subs, test_subs)


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

def get_dataset(subs: dict, mini_batches_per_trial=1, samples_per_mini_batch=64, **kwargs):

    trials = flatten_subs(subs)

    shuffle_samples(trials)

    dset = Trials_Dataset(trials, mini_batches_per_trial=mini_batches_per_trial, samples_per_mini_batch=samples_per_mini_batch)

    return dset

def get_datasets(is_train=True, train_kwargs={}, val_kwargs={}, **kwargs):
    torch.cuda.empty_cache()
    (train_subs, val_subs, test_subs), word_index = load_preprocessed(is_train=is_train, **kwargs)

    if is_train:
        train_dset = get_dataset(train_subs, **kwargs, **train_kwargs)
        val_dset = get_dataset(val_subs, **kwargs, **val_kwargs)
        return train_dset, val_dset, word_index
    else:
        test_dset = get_dataset(test_subs, **kwargs)
        return test_dset, word_index
    
def test_get_datasets():
    train_dset, val_dset, word_index = get_datasets(is_train=True, mini_batches_per_trial=2, samples_per_mini_batch=64)
    print('dset lens: ', len(train_dset), len(val_dset), len(word_index))

if __name__ == '__main__':
    test_get_datasets()
