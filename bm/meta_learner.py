from .state_dict_helpers import average_state_dicts, interpolate_state_dicts
from omegaconf import OmegaConf
import os
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Module
import numpy as np
import logging
import typing as tp
import hydra
from .train import override_args_
from .meta_dataset import get_dataset, get_dataloaders
from copy import deepcopy
from tqdm import trange
from torch.utils.tensorboard import SummaryWriter
import logging
from bm.setup_logging import configure_logging
from bm.meta_dataset2 import get_datasets
from typing import List

base = os.path.dirname(os.path.abspath(__file__))

logger = logging.getLogger(__name__)

seed = 0

# rng = np.random.RandomState(seed)
torch.manual_seed(seed)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train_inner_loop(model: Module, batches, inner_optim=None, n_query=0, inner_lr=0.1):
    """
        query: number of batches to process as query

        output = {"loss_all": forward_output['loss'],
        "clip_loss": forward_output['clip_loss'],
        "clf_loss": forward_output['clf_loss'],
        "mae_loss": forward_output['mae_loss'],
        "focal_loss": forward_output['focal_loss'],
        "ldam_loss": forward_output['ldam_loss'],
        "eeg_embeds": forward_output['eeg_embeds'],
        "text_embeds": forward_output['text_embeds'],
        "clf_logits": forward_output['clf_logits'],}
    """

    old_weights = deepcopy(model.state_dict())

    if not inner_optim:
        inner_optim = optim.SGD(model.parameters(), inner_lr)

    supp_losses = []

    for i in range(len(batches) - n_query):
        # batch = {
        #     'eeg': batches[i]['eeg'].to(DEVICE),
        #     'audio': batches[i]['audio'].to(DEVICE),
        #     'word_index': batches[i]['word_index'].to(DEVICE),
        # }
        output = model.generate(batches[i])

        inner_optim.zero_grad()
        clip_loss: torch.Tensor = output['clip_loss']
        clip_loss.backward()
        inner_optim.step()

        supp_losses.append(clip_loss.item())

    query_losses = []

    with torch.no_grad():
        for i in range(-n_query, 0):
            # batch = {
            #     'eeg': batches[i]['eeg'].to(DEVICE),
            #     'audio': batches[i]['audio'].to(DEVICE),
            #     'word_index': batches[i]['word_index'].to(DEVICE),
            # }
            output = model.generate(batches[i])
            clip_loss: torch.Tensor = output['clip_loss']
            query_losses.append(clip_loss.item())

    new_weights = deepcopy(model.state_dict())

    model.load_state_dict(old_weights)

    return new_weights, supp_losses, query_losses

def meta_train(model: Module, train_loader, val_loader, word_index, meta_optim=None, inner_optim=None, save_dir=None, n_meta_epochs = 1, eval_interval=100, **kwargs):
    """
        Train the ClipMAE-Spatial-Emformer which takes:

        input_sample = {
        'eeg': torch.randn(8, 105, 8), (freq_bins, channels, batch)
        # 'eeg_mask': eeg_mask,
        'label': torch.randint(0, 3, (8,)), (change this to audio)
        'word_id': input_ids,
        'word_attention_mask': attention_mask,
        "epoch": 0,
        "iters": 0,
        "mask_ratio": 0.01,
    }

    """
    writer, checkpoint_dir = setup_logging(save_dir)
    model = model.to(DEVICE)
    
    for meta_epoch in range(n_meta_epochs):
        for i, meta_batch in enumerate(train_loader):
            train_loss = process_meta_batch(model, meta_batch, meta_optim, inner_optim)
            writer.add_scalar('Train loss', train_loss, i)

            if i % eval_interval == 0 or i == len(train_loader) - 1:
                val_loss = evaluate(model, val_loader)
                writer.add_scalar('Val loss', val_loss, i)
                logger.info(f'Meta epoch: {meta_epoch}, meta batch: {i}. Val loss = {val_loss}')
                create_checkpoint(i, model, meta_optim, inner_optim, val_loss, checkpoint_dir)     

def setup_logging(save_dir):
    runs_dir = os.path.join(base, save_dir or 'runs')
    version = 1
    checkpoint_dir = os.path.join(runs_dir, f"v{version}")

    while os.path.exists(checkpoint_dir):
        version += 1
        checkpoint_dir = os.path.join(runs_dir, f"v{version}")

    os.makedirs(checkpoint_dir)
    return SummaryWriter(checkpoint_dir), checkpoint_dir

def create_checkpoint(i, model, meta_optim, inner_optim, loss, checkpoint_dir):
    torch.save({
            'epoch': i,
            'model_state_dict': model.state_dict(),
            'meta_optim_state_dict': meta_optim.state_dict(),
            'inner_optim_state_dict': inner_optim.state_dict(),
            'loss': loss,
            }, os.path.join(checkpoint_dir, f'checkpoint_{i}.pth'))

def evaluate(model, val_loader):
    
    query_losses_all = []

    for meta_batch in val_loader:
        for i in range(meta_batch['eeg'].shape[0]):
            _, _, query_losses = train_inner_loop(model, meta_batch, i, n_query=1)
            query_losses_all.extend(query_losses)

    return sum(query_losses_all) / len(query_losses_all)

def process_meta_batch(model: Module, meta_batch: dict, meta_optim=None, inner_optim=None, meta_lr=0.001):
    weights_before = deepcopy(model.state_dict())
    new_state_dicts = []
    losses = []
    for i in range(len(meta_batch)):
        new_state_dict, supp_losses, _ = train_inner_loop(model, meta_batch[i], inner_optim)
        new_state_dicts.append(new_state_dict)
        losses.extend(supp_losses)
    
    if meta_optim:
        update_params_optim(model, weights_before, new_state_dicts, meta_optim)
    else:
        update_params(model, weights_before, new_state_dicts, meta_lr)
    
    return sum(losses) / len(losses)

def update_params_optim(model: Module, weights_before: dict, new_state_dicts: List[dict], optimizer):
    average_after: dict = average_state_dicts(new_state_dicts)

    optimizer.zero_grad()
    with torch.no_grad():  # Ensure we're not tracking gradients here
        for param, key in zip(model.parameters(), weights_before.keys()):
            # Apply the interpolation update directly
            param.add_(average_after[key] - weights_before[key])

    optimizer.step()


def update_params(model: Module, weights_before, new_state_dicts, meta_lr):
    average_after = average_state_dicts(new_state_dicts)
    
    new_state_dict = interpolate_state_dicts(weights_before, average_after, meta_lr)

    model.load_state_dict(new_state_dict)



class TestModel(Module):
    def __init__(self, num_channels, num_freq, num_classes):
        super(TestModel, self).__init__()

        self.conv1 = torch.nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(16)
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(32)

        self.fc = torch.nn.Linear(32 * num_freq * num_channels, num_classes)
        self.num_classes = num_classes

    def forward(self, x, w_lbs):
        x = x.unsqueeze(1)
        x = F.tanh(self.bn1(self.conv1(x)))
        x = F.tanh(self.bn2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        # x = F.softmax(self.fc(x), dim=-1)
        logits = self.fc(x)
        logits[:, w_lbs]

        return x
    
    def generate(self, x):
        x: dict
        x['eeg'] = x['eeg'].to(DEVICE)
        x['audio'] = x['audio'].to(DEVICE)
        x['w_lbs'] = x['w_lbs'].to(DEVICE).to(torch.int64)
        pred = self.forward(x['eeg'], x['w_lbs'])
        cel = torch.nn.CrossEntropyLoss()
        # one_hot = F.one_hot(x['w_lbs'], num_classes=self.num_classes).to(torch.float32)
        loss = cel(pred, x['w_lbs'])
        return {'clip_loss': loss}

def run(args):
    kwargs: tp.Dict[str, tp.Any]
    kwargs = OmegaConf.to_container(args.dset, resolve=True)  # type: ignore
    selections = [args.selections[x] for x in args.dset.selections]
    kwargs["selections"] = selections
    if args.optim.loss == "clip":
        kwargs['extra_test_features'].append("WordHash")

    # setup_logging(**kwargs)
    meta_dataset = get_dataset(**kwargs)
    train_dataloader = DataLoader(meta_dataset, batch_size=64, shuffle=True)
    dloader = get_dataloaders(meta_dataset)
    model = TestModel()
    return meta_train(model, dloader, **kwargs, num_workers=args.num_workers)

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

def test_meta_train_e2e():
    train_dset, val_dset, word_index = get_datasets(is_train=True, mini_batches_per_trial=1, samples_per_mini_batch=64)
    print('dset lens: ', len(train_dset), len(val_dset), len(word_index))
    train_loader = DataLoader(train_dset, batch_size=2, shuffle=True, collate_fn=lambda x: x)
    val_loader = DataLoader(val_dset, batch_size=2, shuffle=True, collate_fn=lambda x: x)
    model = TestModel(num_channels=208, num_freq=31, num_classes=len(word_index) // 2)
    meta_optim = optim.Adam(model.parameters(), lr=0.001)
    inner_optim = optim.SGD(model.parameters(), lr=0.001)
    return meta_train(model, 
                      train_loader=train_loader, 
                      val_loader=val_loader, 
                      word_index=word_index, 
                      meta_optim=meta_optim, 
                      inner_optim=inner_optim)

def test_meta_train():
    """
        meg for dloader: (trials, batches, meg) 
        meg: (channels, freq)

        audio for dloader: (trials, batches, audio)
        audio: (features1, features2)

        word_index for dloader: (trials, batches, idx)

        dloader: {
            eeg: meg for dloader,
            audio: audio for dloader,
            word_idx: word_indexes for dloader,
        }

        word_index = [...all words]

        we can say that query is a subset of the batches 
        and for train it just uses all the batches
    """
    model = TestModel()
    word_index = ['dog', 'spoon', 'brother', 'dad']                          
    train_dloader = [{
        'eeg': torch.rand((3, 5, 64, 208, 8)),
        'audio': torch.rand((3, 5, 64, 840, 16)),
        'word_index': torch.randint(0, 3, (3, 5, 64)),
    }]
    val_dloader = [{
        'eeg': torch.rand((1, 5, 64, 208, 8)),
        'audio': torch.rand((1, 5, 64, 840, 16)),
        'word_index': torch.randint(0, 3, (1, 5, 64)),
    }]
    meta_optim = optim.Adam(model.parameters(), lr=0.001)
    inner_optim = optim.SGD(model.parameters(), lr=0.001)

    meta_train(model, train_dloader, val_dloader, word_index, meta_optim, inner_optim)

if __name__ == '__main__':
    # main()
    configure_logging()
    # test_meta_train()
    test_meta_train_e2e()