from __future__ import print_function

import argparse
import os

import random
import pandas as pd
from torchvision import datasets, transforms
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
import numpy as np

import time
import math
from torch.multiprocessing import Process


WORLD_SIZE = int(os.environ.get('WORLD_SIZE', 1))

class IDSDataset(Dataset):
    def __init__(self):
      df = pd.read_csv('gs://kbc/churn/output/train.csv')
      df_labels = pd.read_csv('gs://kbc/churn/output/train_labels.csv')
      df = df.iloc[1:, :]
      df_labels = df_labels.iloc[1:, :]
      self.trainX = df.to_numpy().astype(np.float32)
      self.trainy = df_labels.to_numpy().astype(np.int64).flatten()
      self.n_samples = self.trainX.shape[0]

    def __getitem__(self, index):
      return self.trainX[index], self.trainy[index]

    def __len__(self):
      return self.n_samples  

class Partition(object):

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


class DataPartitioner(object):

    def __init__(self, data, sizes=[0.6, 0.2, 0.2], seed=1234):
        self.data = data
        self.partitions = []
        random.seed(seed)
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        random.shuffle(indexes)

        for frac in sizes:
            part_len = int(frac * data_len)
            print('Length of partition is', part_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #self.fc1 = nn.Linear(79, 32)
        #self.fc2 = nn.Linear(32, 16)
        #self.fc3 = nn.Linear(16, 3)
        self.fc1 = nn.Linear(79, 8)
        self.fc2 = nn.Linear(8, 3)

    def forward(self, x):
        x = x.view(-1, 79)
        x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def average_gradients(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= size

def partition_dataset(dataset):
    size = dist.get_world_size()
    bsz = int(1024 / int(size))
    partition_sizes = [1.0 / size for _ in range(size)]
    partition = DataPartitioner(dataset, partition_sizes)
    partition = partition.use(dist.get_rank())
    train_set = torch.utils.data.DataLoader(partition,
                                         batch_size=bsz,
                                         shuffle=True)
    return train_set, bsz        
    
def train(args, model, device, train_loader, batch_size, optimizer, epoch):
    model.train()
    num_batches = math.ceil(len(train_loader.dataset) / float(batch_size))
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        if  not math.isnan(loss):
            loss.backward()
        optimizer.step()
        average_gradients(model)
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tloss={:.4f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(args, model, device, test_loader, writer, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\naccuracy={:.4f}\n'.format(float(correct) / len(test_loader.dataset)))
    writer.add_scalar('accuracy', float(correct) / len(test_loader.dataset), epoch)

""" Gradient averaging. """
def average_gradients(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= size

def should_distribute():
    return dist.is_available() and WORLD_SIZE > 1


def is_distributed():
    return dist.is_available() and dist.is_initialized()


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='IDS Cisco')
    parser.add_argument('--batch-size', type=int, default=1024, metavar='N',
                        help='input batch size for training (default: 1024)')
    parser.add_argument('--test-batch-size', type=int, default=1024, metavar='N',
                        help='input batch size for testing (default: 1024)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=True,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--dir', default='logs', metavar='L',
                        help='directory where summary logs are stored')
    if dist.is_available():
        parser.add_argument('--backend', type=str, help='Distributed backend',
                            choices=[dist.Backend.GLOO, dist.Backend.NCCL, dist.Backend.MPI],
                            default=dist.Backend.GLOO)
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if use_cuda:
        print('Using CUDA')


    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    if should_distribute():
        print('Using distributed PyTorch with {} backend'.format(args.backend))
        dist.init_process_group(backend=args.backend)

    model = Net().to(device)

    if is_distributed():
        Distributor = nn.parallel.DistributedDataParallel if use_cuda \
            else nn.parallel.DistributedDataParallelCPU
        model = Distributor(model)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    dataset = IDSDataset()
    
    train_loader, batch_size = partition_dataset(dataset)    
    
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, batch_size, optimizer, epoch)
        #test(args, model, device, test_loader, writer, epoch)

    if (args.save_model):
        torch.save(model.state_dict(),"ids.pt")
        
if __name__ == '__main__':
    main()
