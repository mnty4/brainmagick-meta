import numpy as np
import torch
import os
import glob
from typing import Tuple, List
import regex as re
# base = os.path.abspath(__package__)
base = "/projects/SilSpeech/Dev/SilentSpeech_Se2/listen_meg_eeg_preprocess/brainmagick/bm/" 
def load_preprocessed(is_train=True, seed=42, **kwargs) -> Tuple[dict, dict]:
    np.random.seed(seed)
    save_path = os.path.join(base, 'preprocessed')
    tensor_paths = glob.glob(os.path.join(save_path, '*.pt'))
    print('save_path',save_path,'tensor_paths',tensor_paths)
    sub_count = sum(1 for tensor_path in tensor_paths if re.match('sub-\d+\.pt', os.path.basename(tensor_path)))
    print('sub_count',sub_count)
    train_ids, val_ids, test_ids = split_subs(sub_count)
    print('train_ids',train_ids,'val_ids',val_ids,'test_ids',test_ids)

    subs = {}
    word_index = {}
    for tensor_path in tensor_paths:
        print('loading tensor_path',tensor_path)
        assert os.path.exists(tensor_path)
        name = os.path.basename(tensor_path)
        if name == 'word_index.pt':
            word_index = torch.load(tensor_path, weights_only=True)
        else:
            subs[name] = torch.load(tensor_path, weights_only=True)
    return subs, word_index

def split_subs(sub_count: int, **kwargs) -> Tuple[List[int], List[int], List[int]]:
    assert sub_count > 0

    if sub_count == 1:
        return ([0], [0], [0])
    
    if sub_count == 2:
        return ([0], [1], [1])

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

    
def get_dataset(**kwargs):
    load_preprocessed(**kwargs)


if __name__ == '__main__':
    load_preprocessed()