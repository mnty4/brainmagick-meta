from lavis import registry
# import lavis.models.belt3_models.belt_clip_mae import Clip_TemporalConformer2D
import lavis.models.belt3_models.belt_clip_mae
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
from bm.models.classification_head import EEG_Encoder_Classification_Head
from bm.meta_evaluate import test
base = os.path.dirname(os.path.abspath(__file__))

logger = logging.getLogger(__name__)

seed = 0

# rng = np.random.RandomState(seed)
torch.manual_seed(seed)
np.random.seed(seed)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def train_inner_loop(model: Module, batches, inner_optim=None, n_query=0, inner_lr=0.1, loss_type='clip_loss'):
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
    model.train()
    for i in range(len(batches) - n_query):
        # batches = {
        #     'eeg': batches[i]['eeg'].to(DEVICE),
        #     'audio': batches[i]['audio'].to(DEVICE),
        #     'word_index': batches[i]['word_index'].to(DEVICE),
        # }
        batches[i]['eeg'] = batches[i]['eeg'].to(DEVICE)
        batches[i]['audio'] = batches[i]['audio'].to(DEVICE)
        batches[i]['w_lbs'] = batches[i]['w_lbs'].to(DEVICE)
        output = model.generate(batches[i])

        inner_optim.zero_grad()
        loss: torch.Tensor = output[loss_type]
        loss.backward()
        inner_optim.step()

        supp_losses.append(loss.item())

    query_losses = []
    model.eval()
    with torch.no_grad():
        for i in range(-n_query, 0):
            batches[i]['eeg'] = batches[i]['eeg'].to(DEVICE)
            batches[i]['audio'] = batches[i]['audio'].to(DEVICE)
            batches[i]['w_lbs'] = batches[i]['w_lbs'].to(DEVICE)
            output = model.generate(batches[i])
            loss: torch.Tensor = output[loss_type]
            query_losses.append(loss.item())

    new_weights = deepcopy(model.state_dict())

    model.load_state_dict(old_weights)

    return new_weights, supp_losses, query_losses

def meta_train(model: Module, train_loader, val_loader, word_index, meta_optim=None, inner_optim=None, save_dir=None, n_meta_epochs = 1, eval_interval=100, do_meta_train=True, early_stop_patience=5, delta = 0.05, train_type='clip', **kwargs):
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

    writer, checkpoint_dir = setup_logging(save_dir, do_meta_train=do_meta_train, train_type=train_type, **kwargs)

    model = model.to(DEVICE)
    best_model = None
    best_val_loss, best_train_loss = float('inf'), float('inf')
    best_epoch, best_i = None, None
    for meta_epoch in range(n_meta_epochs):
        for i, meta_batch in enumerate(train_loader):
            train_loss = process_meta_batch(model, meta_batch, meta_optim, inner_optim, do_meta_train=do_meta_train, **kwargs)
            writer.add_scalar('Train loss', train_loss, meta_epoch*len(train_loader) + i)

            if i % eval_interval == 0 or i == len(train_loader) - 1:
                val_loss = evaluate(model, val_loader, inner_optim, **kwargs)
                writer.add_scalar('Val loss', val_loss, meta_epoch*len(train_loader) + i)
                logger.info(f'Meta epoch: {meta_epoch}, meta batch: {i}. Train loss = {train_loss}, Val loss = {val_loss}')
                create_checkpoint(meta_epoch, i, model, meta_optim, inner_optim, val_loss, checkpoint_dir)  
                if i == len(train_loader) - 1 and val_loss < best_val_loss - delta:
                    best_val_loss = val_loss
                    best_train_loss = train_loss
                    best_model = deepcopy(model.state_dict())
                    best_epoch = meta_epoch
                    best_i = i

        if meta_epoch - best_epoch >= early_stop_patience:
            logger.info('Early stopping triggered...')
            break
    
    logger.info(f'[Best model] Meta epoch: {best_epoch}, meta batch: {best_i}. Train loss = {best_train_loss}, Val loss = {best_val_loss}')
    writer.close()

    old_path = os.path.join(checkpoint_dir, f'checkpoint_{best_epoch}_{best_i}.pth')
    best_path = os.path.join(checkpoint_dir, f'best_checkpoint_{best_epoch}_{best_i}.pth')
    os.rename(old_path, best_path)

    model.load_state_dict(best_model)
    return best_path

def setup_logging(save_dir, do_meta_train=True, train_type='clip', do_meta_train_for_head=False, **kwargs):
    runs_dir = os.path.join(base, save_dir or 'runs')
    version = 1
    file_name = f"{'non_meta_' if not do_meta_train else 'meta_'}"
    if train_type == 'classifier':
        file_name = 'm_head_' if do_meta_train_for_head else 'nm_head_' + 'classifier_' + file_name

    checkpoint_dir = os.path.join(runs_dir, f"{file_name}v{version}")

    while os.path.exists(checkpoint_dir):
        version += 1
        checkpoint_dir = os.path.join(runs_dir, f"{file_name}v{version}")
    
    os.makedirs(checkpoint_dir)
    return SummaryWriter(checkpoint_dir), checkpoint_dir

def create_checkpoint(meta_epoch, i, model, meta_optim, inner_optim, loss, checkpoint_dir):
    torch.save({
            'meta_epoch': meta_epoch,
            'batch': i,
            'is_meta_train': True if meta_optim else False,
            'model_state_dict': model.state_dict(),
            'meta_optim_state_dict': meta_optim.state_dict() if meta_optim else None,
            'inner_optim_state_dict': inner_optim.state_dict(),
            'loss': loss,
            }, os.path.join(checkpoint_dir, f'checkpoint_{meta_epoch}_{i}.pth'))

def evaluate(model, val_loader, inner_optim=None, **kwargs):
    
    query_losses_all = []

    for meta_batch in val_loader:
        for i in range(len(meta_batch)):
            _, _, query_losses = train_inner_loop(model, meta_batch[i], inner_optim, n_query=1, **kwargs)
            query_losses_all.extend(query_losses)

    return sum(query_losses_all) / len(query_losses_all)

def process_meta_batch(model: Module, meta_batch: dict, meta_optim=None, inner_optim=None, do_meta_train=True, meta_lr=0.01, **kwargs):
    weights_before = deepcopy(model.state_dict())
    new_state_dicts = []
    losses = []
    for i in range(len(meta_batch)):
        new_state_dict, supp_losses, _ = train_inner_loop(model, meta_batch[i], inner_optim, **kwargs)
        if not do_meta_train:
            model.load_state_dict(new_state_dict)
        else:
            new_state_dicts.append(new_state_dict)
        losses.extend(supp_losses)
    
    if do_meta_train:
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
    registry
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

def train_combined_clf(train_kwargs, val_kwargs, test_kwargs, do_meta_train=False, n_meta_epochs=20, inner_lr=0.001, meta_lr=0.001, loss_type='combined_loss', **kwargs):
    train_kwargs = {'mini_batches_per_trial': 1, 'samples_per_mini_batch': 128, 'batch_size': 2, **train_kwargs}
    val_kwargs = {'mini_batches_per_trial': 2, 'samples_per_mini_batch': 128, 'batch_size': 2, **val_kwargs}

    train_dset, val_dset, word_index = get_datasets(is_train=True, train_kwargs=train_kwargs, val_kwargs=val_kwargs)
    print('dset lens: ', len(train_dset), len(val_dset), len(word_index))
    train_loader = DataLoader(train_dset, batch_size=train_kwargs['batch_size'], shuffle=True, collate_fn=lambda x: x)
    val_loader = DataLoader(val_dset, batch_size=val_kwargs['batch_size'], shuffle=True, collate_fn=lambda x: x)

    model_cls = registry.get_model_class("Clip-Audio-Emformer")    
    model = model_cls.from_config(type='base', num_classes = len(word_index) // 2)
    if do_meta_train:
        meta_optim = optim.Adam(model.parameters(), lr=meta_lr)
        inner_optim = optim.SGD(model.parameters(), lr=inner_lr)
    else:
        meta_optim = None
        inner_optim = optim.Adam(model.parameters(), lr=inner_lr)

    inner_optim = optim.Adam(model.parameters(), lr=inner_lr)
    best_model_path = meta_train(model, 
                      train_loader=train_loader, 
                      val_loader=val_loader, 
                      word_index=word_index, 
                      meta_optim=meta_optim, 
                      inner_optim=inner_optim,
                      n_meta_epochs=n_meta_epochs,
                      do_meta_train=do_meta_train,
                      loss_type=loss_type)
    
    ks = [1, 5, 15, 50, 500, 1500]
    del model
    del model_cls
    del train_loader
    del val_loader
    del word_index
    del inner_optim
    
    test(best_model_path, seed=42, ks=ks, type='classifier_combined', loss_type=loss_type, **test_kwargs)



def train_clf_head(train_kwargs, val_kwargs, test_kwargs, do_meta_train=False, do_meta_train_for_head=False, n_meta_epochs=20, inner_lr=0.001, meta_lr=0.001, **kwargs):
    train_kwargs = {'mini_batches_per_trial': 1, 'samples_per_mini_batch': 128, 'batch_size': 2, **train_kwargs}
    val_kwargs = {'mini_batches_per_trial': 2, 'samples_per_mini_batch': 128, 'batch_size': 2, **val_kwargs}

    train_dset, val_dset, word_index = get_datasets(is_train=True, train_kwargs=train_kwargs, val_kwargs=val_kwargs)
    print('dset lens: ', len(train_dset), len(val_dset), len(word_index))
    train_loader = DataLoader(train_dset, batch_size=2, shuffle=True, collate_fn=lambda x: x)
    val_loader = DataLoader(val_dset, batch_size=2, shuffle=True, collate_fn=lambda x: x)

    model_cls = registry.get_model_class("Clip-Audio-Emformer")    
    model = model_cls.from_config(type='base', use_classifier=False)
    if do_meta_train:
        meta_optim = optim.Adam(model.parameters(), lr=meta_lr)
        inner_optim = optim.SGD(model.parameters(), lr=inner_lr)
    else:
        meta_optim = None
        inner_optim = optim.Adam(model.parameters(), lr=inner_lr)
    
    best_model_path = meta_train(model, 
                      train_loader=train_loader, 
                      val_loader=val_loader, 
                      word_index=word_index, 
                    #   meta_optim=meta_optim, 
                      inner_optim=inner_optim,
                      n_meta_epochs=n_meta_epochs,
                      do_meta_train=do_meta_train,
                      loss_type='clip_loss')

    classification_head = EEG_Encoder_Classification_Head(
        model.eeg_encoder, num_classes=len(word_index) // 2, eeg_projection=model.eeg_projection)
    
    inner_classifier_optim = optim.Adam(classification_head.parameters(), lr=0.001)

    best_classifier_model_path = meta_train(classification_head,
                    #  save_dir=best_model_path,
                       train_loader=train_loader, 
                      val_loader=val_loader, 
                      word_index=word_index, 
                    #   meta_optim=meta_optim, 
                      inner_optim=inner_classifier_optim,
                      n_meta_epochs=n_meta_epochs,
                      do_meta_train=do_meta_train_for_head,
                      train_type='classifier',
                      loss_type='ce_loss')

    del model
    del model_cls
    del train_loader
    del val_loader
    del inner_optim
    
    ks = [1, 5, 15, 50, 500, 1500]
    test(best_classifier_model_path, seed=42, ks=ks, type='classifier_head', loss_type='ce_loss', **test_kwargs)


# def test_meta_train_combined_clf():
#     train_kwargs = {'mini_batches_per_trial': 1, 'samples_per_mini_batch': 128}
#     val_kwargs = {'mini_batches_per_trial': 2, 'samples_per_mini_batch': 128}
#     train_dset, val_dset, word_index = get_datasets(is_train=True, train_kwargs=train_kwargs, val_kwargs=val_kwargs)
#     print('dset lens: ', len(train_dset), len(val_dset), len(word_index))
#     train_loader = DataLoader(train_dset, batch_size=2, shuffle=True, collate_fn=lambda x: x)
#     val_loader = DataLoader(val_dset, batch_size=2, shuffle=True, collate_fn=lambda x: x)
#     # model = TestModel(num_channels=208, num_freq=31, num_classes=len(word_index) // 2)
#     model_cls = registry.get_model_class("Clip-Audio-Emformer")    
#     model = model_cls.from_config(type='base') #.cpu()
#     meta_optim = optim.Adam(model.parameters(), lr=0.001)
#     inner_optim = optim.SGD(model.parameters(), lr=0.001)
#     best_model_path = meta_train(model, 
#                       train_loader=train_loader, 
#                       val_loader=val_loader, 
#                       word_index=word_index, 
#                       meta_optim=meta_optim, 
#                       inner_optim=inner_optim,
#                       n_meta_epochs=20)
    
#     ks = [1, 5, 15, 50, 500, 1500]
#     test(best_model_path, seed=42, ks=ks, type='clip')

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

    meta_train(model, train_dloader, val_dloader, word_index, meta_optim, inner_optim, n_meta_epochs=20)

def run_tests():
    # n_meta_epochs=20, inner_lr=0.001, meta_lr=0.001,

    # train_kwargs = {
    #     'mini_batches_per_trial': 1, 
    #     'samples_per_mini_batch': 128, 
    #     'batch_size': 2,
    # }
    # val_kwargs = {
    #     'mini_batches_per_trial': 1, 
    #     'samples_per_mini_batch': 128, 
    #     'batch_size': 2,
    # }
    # test_kwargs = {
    #     'mini_batches_per_trial': 1, 
    #     'samples_per_mini_batch': 128, 
    #     'batch_size': 2,
    # }
    # test_kwargs['model_name'] = 'no_meta_combined_clf_2_1_128_shot_0_128'
    # train_combined_clf(train_kwargs, val_kwargs, test_kwargs, do_meta_train=False)
    # test_kwargs['model_name'] = 'no_meta_clf_head_2_1_128_shot_0_128'
    # train_clf_head(train_kwargs, val_kwargs, test_kwargs, do_meta_train=False)
    # test_kwargs['model_name'] = 'meta_combined_clf_2_1_128_shot_0_128'
    # train_combined_clf(train_kwargs, val_kwargs, test_kwargs, do_meta_train=True)
    # test_kwargs['model_name'] = 'meta_clf_nmhead_2_1_128_shot_0_128'
    # train_clf_head(train_kwargs, val_kwargs, test_kwargs, do_meta_train=True)
    # test_kwargs['model_name'] = 'meta_clf_mhead_2_1_128_shot_0_128'
    # train_clf_head(train_kwargs, val_kwargs, test_kwargs, do_meta_train=True, do_meta_train_for_head=True)

    train_kwargs = {
        'mini_batches_per_trial': 1, 
        'samples_per_mini_batch': 64, 
        'batch_size': 4,
    }
    val_kwargs = {
        'mini_batches_per_trial': 5, 
        'samples_per_mini_batch': 8, 
        'batch_size': 4,
    }
    test_kwargs = {
        'mini_batches_per_trial': 8, 
        'samples_per_mini_batch': 8, 
        'batch_size': 1,
        'unfreeze_encoder_on_support': False,
        'n_shot': 4,
    }
    kwargs = {
        'n_meta_epochs': 5
    }
    test_kwargs['model_name'] = 'no_meta_combined_clf_4_4_8_shot_4_8'
    train_combined_clf(train_kwargs, val_kwargs, test_kwargs, do_meta_train=False, **kwargs)
    test_kwargs['model_name'] = 'no_meta_clf_head_4_4_8_shot_4_8'
    train_clf_head(train_kwargs, val_kwargs, test_kwargs, do_meta_train=False, **kwargs)
    test_kwargs['model_name'] = 'meta_combined_clf_4_4_8_shot_4_8'
    train_combined_clf(train_kwargs, val_kwargs, test_kwargs, do_meta_train=True, **kwargs)
    test_kwargs['model_name'] = 'meta_clf_nmhead_4_4_8_shot_4_8'
    train_clf_head(train_kwargs, val_kwargs, test_kwargs, do_meta_train=True, **kwargs)
    test_kwargs['model_name'] = 'meta_clf_mhead_4_4_8_shot_4_8'
    train_clf_head(train_kwargs, val_kwargs, test_kwargs, do_meta_train=True, do_meta_train_for_head=True, **kwargs)

    # test_kwargs['unfreeze_encoder_on_support'] = True

    # test_kwargs['model_name'] = 'meta_clf_nmhead_4_4_8_shot_4_8_unfreeze'
    # train_clf_head(train_kwargs, val_kwargs, test_kwargs, do_meta_train=True)
    # test_kwargs['model_name'] = 'meta_clf_mhead_4_4_8_shot_4_8_unfreeze'
    # train_clf_head(train_kwargs, val_kwargs, test_kwargs, do_meta_train=True, do_meta_train_for_head=True)

if __name__ == '__main__':
    # main()
    configure_logging()
    run_tests()
    # test_meta_train()
    # test_meta_train_e2e()
    # test_normal_train_e2e()
    # test_normal_train_combined_clf()