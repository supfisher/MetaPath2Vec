from read_data import ReadData, Copus
from torch.utils.data import DataLoader
from SGNS import SGNS
from torch.optim import Adam, lr_scheduler
from tqdm import tqdm
import torch as t
import time
import pickle
import numpy as np
import os
import argparse
import torch.distributed as dist
import torch.nn as nn
import torch.multiprocessing as mp
import sys


parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='sgns', help="model name")
parser.add_argument('--data_dir', type=str, default='./data/', help="data directory path")
parser.add_argument('--save_dir', type=str, default='./pts/', help="model directory path")
parser.add_argument('--e_dim', type=int, default=300, help="embedding dimension")
parser.add_argument('--n_negs', type=int, default=20, help="number of negative samples")
parser.add_argument('--epoch', type=int, default=100, help="number of epochs")
parser.add_argument('--mb', type=int, default=4096, help="mini-batch size")
parser.add_argument('--ss_t', type=float, default=1e-5, help="subsample threshold")
parser.add_argument('--conti', action='store_true', help="continue learning")
parser.add_argument('--weights', default=True, action='store_true', help="use weights for negative sampling")
parser.add_argument('--cuda', action='store_true', help="use CUDA")
parser.add_argument('-b', '--batch-size', default=20480, type=int)
parser.add_argument('--num_workers', default=4, type=int,
                    help='number of workers')


def main_worker(args):
    device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
    beginning = time.time()

    read_data = ReadData("/ibex/scratch/mag0a/Github/data/aminer.txt")
    print("read file cost time: ", time.time() - beginning)
    # dataset = Copus(read_data)

    dataset = Copus("./data")
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    print("dataloader cost time: ", time.time() - beginning)

    idx2word = read_data.idx2word
    wc = read_data.word_count
    wf = np.array([wc[word] for word in idx2word])
    wf = wf / wf.sum()
    weights = t.tensor(wf) if args.weights else None
    if weights is not None:
        wf = t.pow(weights, 0.75)
        weights = (wf / wf.sum()).float()

    model = SGNS(100000, 128, n_negs=20)
    model = nn.DataParallel(model)
    model.to(device)

    optimizer = Adam(model.parameters(), lr=0.025)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, len(dataloader))

    print("training preperation cost time: ", time.time() - beginning)
    model.train()
    for epoch in range(4):
        for i, (u, v) in enumerate(tqdm(dataloader)):
            u, v, weights = u.to(device), v.to(device), weights.to(device)
            optimizer.zero_grad()
            loss = model(u, v, weights)
            loss = loss.mean()
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss = loss.item()
            if i > 0 and i % 1000 == 0:
                print(" Loss: " + str(running_loss))

        t.save(model.state_dict(), 'model_%s.pkl' % epoch)


if __name__ == "__main__":
    args = parser.parse_args()
    ngpus = t.cuda.device_count()
    print("number of used GPUs: ", ngpus)
    main_worker(args)
