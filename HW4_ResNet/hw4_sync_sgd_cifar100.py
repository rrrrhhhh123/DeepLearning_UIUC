# -*- coding: utf-8 -*-
"""sync_sgd_cifar100.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/11EpK-fFEPqO9JlA7icjsaD37WSg04OsJ
"""

import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.distributed as dist
import os
import subprocess
from mpi4py import MPI
import MyResNet

# Code for iniitialization pytorch distributed
cmd = "/sbin/ifconfig"
out, err = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE).communicate()
ip = str(out).split("inet addr:")[1].split()[0]

name = MPI.Get_processor_name()
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
num_nodes = int(comm.Get_size())

ip = comm.gather(ip)

if rank != 0:
  ip = None

ip = comm.bcast(ip, root=0)

os.environ['MASTER_ADDR'] = ip[0]
os.environ['MASTER_PORT'] = '2222'

backend = 'mpi'
dist.init_process_group(backend, rank=rank, world_size=num_nodes)

dtype = torch.FloatTensor



# hyperparameters
num_epochs = 50
batch_size = 128
learning_rate = 0.001

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


trainset = torchvision.datasets.CIFAR100(root='./', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR100(root='./', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

def calculate_accuracy(data_loader):
    """
        Calculate the model's predict accuracy on data set
    """
    correct = 0
    total = 0
    with torch.no_grad():
        model.eval()
        for batch_idx, (X_test_batch, Y_test_batch) in enumerate(data_loader):
            X_test_batch, Y_test_batch = X_test_batch.cuda(),Y_test_batch.cuda() # old version of pytorch(0.3)
            
            outputs = model(X_test_batch)
            _, predicted = torch.max(outputs.data, 1)
            total += Y_test_batch.size(0)
            correct += (predicted == Y_test_batch).sum().item()

    return correct/total

""" Gradient averaging. """
def average_gradients(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
        param.grad.data /= size

def run(rank, size):

    torch.manual_seed(1234)

    model = MyResNet.MyResNet(MyResNet.BasicBlock, [2, 4, 4, 2], 100, 3, 32).cuda()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    if (rank == 0):
        acc_df = pd.DataFrame(columns=["Train Accuracy", "Test Accuracy"])

    for epoch in range(num_epochs):

        epoch_loss = 0.0
        model.train()
        for batch_idx, (X_train_batch, Y_train_batch) in enumerate(trainloader):
            X_train_batch, Y_train_batch = X_train_batch.cuda(), Y_train_batch.cuda()
            optimizer.zero_grad()
            
            # Forward Pass
            outputs = model(X_train_batch)
            loss = criterion(outputs, Y_train_batch)
            epoch_loss += loss.item()

            # Backward and optimize
            loss.backward()
            average_gradients(model)

            if (epoch >= 6):
                for group in optimizer.param_groups:
                    for p in group['params']:
                        state = optimizer.state[p]
                        if 'state' in state.keys():
                            if (state['step'] >= 1024):
                                state['step'] = 1000

            optimizer.step()
            
        print('Rank ', dist.get_rank(), ', epoch ', epoch, ': ', epoch_loss)

        if (rank == 0):
            train_acc, test_acc = calculate_accuracy(trainloader), calculate_accuracy(testloader)
            print(', Train Accuracy: ', train_acc,
                    ', Test Accuracy: ', test_acc)
            # save the accuracy for this epoch
            acc_df = acc_df.append({"Train Accuracy": train_acc,
                                    "Test Accuracy": test_acc},
                                ignore_index=True)
            acc_df.to_csv("./accuracy_sync_sgd_cifar100.csv")

if __name__ == '__main__':
    run(dist.get_rank(), num_nodes)