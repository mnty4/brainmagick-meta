from omegaconf import OmegaConf
import os
import torch
from torch.autograd import Variable
import numpy as np
import logging
import typing as tp
import hydra
from .train import override_args_
from .meta_dataset import get_datasets

logger = logging.getLogger(__name__)


file = open("Output_subj3.txt", "w")
np.random.seed(42)
torch.manual_seed(42)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
file.write(f"DEVICE = {DEVICE}\n")


def train(model, x_sup, y_sup, x_qry, y_qry):
    for x, y in zip(x_sup, y_sup):
        y_pred = model(x)
        loss(y_pred, y)
        step
    
    total_loss = 0
    for x, y in zip(x_qry, y_qry):
        freeze
        y_pred = model(x)
        total_loss += loss(y_pred, y)
    
    return total_loss

def get_batch():
    pass

def eval():
    pass

def train(model, epochs, eval_every=50):
    # 32 x 208 x F
    negative_set = []
    for epoch in range(epochs):
        x_sup, y_sup, x_qry, y_qry = get_batch()
        output = model(x_sup, y_sup, x_qry, y_qry)

        if epoch % eval_every == 0:
            results = eval()
            log.


    for x, y in zip(x_sup, y_sup):
        y_pred = model(x)
        y_preds.append(y_pred)
        loss = CLIP(y_pred, y)

        step(loss)
    
    total_loss = 0
    for x, y in zip(x_qry, y_qry):
        freeze
        y_pred = model(x)
        total_loss += loss(y_pred, y)
    
    return total_loss


def meta_train(db, meta, iterations):
    file.write("\nReptile\n")
    for episode_num in range(iterations):
        support_x, support_y, query_x, query_y = db.get_batch('train')
        support_x = Variable( torch.from_numpy(support_x).float()).to(DEVICE)
        query_x = Variable( torch.from_numpy(query_x).float()).to(DEVICE)
        support_y = Variable(torch.from_numpy(support_y).long()).to(DEVICE)
        query_y = Variable(torch.from_numpy(query_y).long()).to(DEVICE)

        accs = meta(support_x, support_y, query_x, query_y)
        train_acc = 100 * np.array(accs).mean()

        if episode_num % 50 == 0:
            test_accs = []
            for i in range(min(episode_num // 5000 + 3, 10)):
                support_x, support_y, query_x, query_y = db.get_batch('test')
                support_x = Variable( torch.from_numpy(support_x).float()).to(DEVICE)
                query_x = Variable( torch.from_numpy(query_x).float()).to(DEVICE)
                support_y = Variable(torch.from_numpy(support_y).long()).to(DEVICE)
                query_y = Variable(torch.from_numpy(query_y).long()).to(DEVICE)

                test_acc = meta.pred(support_x, support_y, query_x, query_y)
                test_accs.append(test_acc)

            test_acc = 100 * np.array(test_accs).mean()
            file.write(f"episode: {episode_num}\tfinetune acc: {train_acc:.5f}\t\ttest acc: {test_acc:.5f}\n")


def train(net, train_loader, epochs):
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    train_log = []
    val_log = []
    for epoch in range(epochs):
        accs = []
        train_loss = []
        val_loss = []
        for support_x, support_y, query_x, query_y in tqdm(train_loader):
            support_x = Variable(support_x[0].float()).to(DEVICE)
            query_x = Variable(query_x[0].float()).to(DEVICE)
            support_y = Variable(support_y[0].long()).to(DEVICE)
            query_y = Variable(query_y[0].long()).to(DEVICE)

            net.train()
            loss, pred = net(support_x, support_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

            net.eval()
            loss, pred = net(query_x, query_y)
            val_loss.append(loss.item())
            indices = torch.argmax(pred, dim=1)
            correct = torch.eq(indices, query_y).sum().item()
            acc = correct / query_y.size(0)
            accs.append(acc)
        train_loss = np.array(train_loss).mean()
        train_log.append(train_loss)
        val_loss = np.array(val_loss).mean()
        val_log.append(val_loss)
        accuracy = 100 * np.array(accs).mean()
        file.write(f"Epoch {epoch+1}: \tvalidation acc: {accuracy:.5f}\tvalidation loss: {val_loss:.6f}\ttrain loss: {train_loss:.6f}\n")
    plt.plot(train_log)
    plt.plot(val_log)
    plt.show()

# def setup_logging(logging_level, log_dir, **kwargs):
#     logging.basicConfig(level=logging_level, format='%(asctime)s [%(levelname)s] - %(module)s: %(message)s', handlers=[
# 																logging.StreamHandler(), 
# 																logging.FileHandler(os.path.join(log_dir, 'games_download.log'), mode='w')])
 
def run(args):
    kwargs: tp.Dict[str, tp.Any]
    kwargs = OmegaConf.to_container(args.dset, resolve=True)  # type: ignore
    selections = [args.selections[x] for x in args.dset.selections]
    kwargs["selections"] = selections
    if args.optim.loss == "clip":
        kwargs['extra_test_features'].append("WordHash")

    # setup_logging(**kwargs)
    get_datasets(**kwargs)
    return train(**kwargs, num_workers=args.num_workers)

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

if __name__ == '__main__':
    main()