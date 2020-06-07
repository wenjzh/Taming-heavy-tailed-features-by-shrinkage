# This script implements numerical study on MNIST dataset. The main idea is to compare
# the classification performance from original features and shrunk features.

# Author: Wenjing Zhou, Ziwei Zhu
# Last modified date: 06/07/2020

import warnings
from numpy import linalg as LA
import numpy as np
import random
import math
import multiprocessing
from scipy import stats
from statistics import mean
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch.nn as nn
import torchvision
import torch
MISLABLE_RATE = 0.4
ILIST = range(2)
THRESHOLD = 2
NUM_EPOCHS_CNN = 2
NUM_EPOCHS_LOG = 1
BATCH_SIZE_TRAIN = 117
BATCH_SIZE_TEST = 181
LEARNING_RATE = 1e-5

warnings.filterwarnings("ignore")

def changeto_01_train(dataset):
    '''
    This function changes the train_labels from 4/9 to 0/1
    @input dataset: MINST tranining dataset from torchvision
    @return dataset: modified training dataset
    '''

    for i in list(idx_train[0]):
        if dataset.train_labels[i] == 4:
            dataset.train_labels[i] = 0
        else:
            dataset.train_labels[i] = 1
    return dataset


def changeto_01_test(dataset):
    '''
    This function changes the test_labels from 4/9 to 0/1
    @input dataset: MINST testing dataset from torchvision
    @return dataset: modified testing dataset
    '''

    for i in list(idx_test[0]):
        if dataset.test_labels[i] == 4:
            dataset.test_labels[i] = 0
        else:
            dataset.test_labels[i] = 1
    return dataset


def flip(label):
    if label == 4:
        x = 9
    else:
        x = 4
    return x


def flip_train(mislable_rate, dataset):
    '''
    This function corrupts the lables of the training dataset
    @input dataset: MINST training dataset from torchvision
    @param mislable_rate: the probability of mislableing
    @return dataset: mislabled training dataset
    '''

    size = math.floor(len(idx_train[0]) * mislable_rate)
    flip_index = random.sample(list(idx_train[0]), size)
    for i in flip_index:
        dataset.train_labels[i] = flip(dataset.train_labels[i])
    return dataset


class ConvNet(nn.Module):

    ''' Original CNN'''

    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=5,
                stride=1,
                padding=2),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2))
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(7 * 7 * 64, 1000)
        self.fc2 = nn.Linear(1000, 2)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)

        # Features learned
        out_x = self.fc1(out)

        out = self.fc2(out_x)
        return out, out_x


class LogisticRegression(torch.nn.Module):

    '''Logistic regression'''

    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(1000, 2)

    def forward(self, x):
        outputs = self.linear(x)
        return outputs


sm = nn.Softmax()
nloss = nn.NLLLoss()


def weighted_nll(input, target, mislable_rate):
    '''
    This function implements the weighted negative likelihood
    @input input: tensor
    @input target: tensor
    @param mislable_rate: the probability of mislabling
    @return : the weighted negative likelihood
    '''

    l1 = nloss(F.log_softmax(input), target)
    l2 = nloss(torch.log(1 - sm(input)), target)
    return ((1 - mislable_rate) * l1 - mislable_rate * l2) / \
        (1 - 2 * mislable_rate)


def train1(i, epoch, mislable_rate, loader):
    '''
    This function trains the original CNN
    '''

    for j, (images, labels) in enumerate(loader):
        out, _ = model1[i](images)
        loss = weighted_nll(out, labels, mislable_rate)

        optimizer1[i].zero_grad()
        loss.backward()
        optimizer1[i].step()

        total = labels.size(0)
        _, predicted = torch.max(out.data, 1)
        correct = (predicted == labels).sum().item()
    print('Cnn: Epoch [{}/{}], Loss: {:.4f}, Training Accuracy: {:.2f}%'.format(
        epoch + 1, NUM_EPOCHS_CNN, loss.item(), (correct / total) * 100))


def test1(i):
    '''
    This function tests the original CNN
    '''

    model1[i].eval()
    with torch.no_grad():  # deactivate autograd engine to reduce memory usage and speed up
        correct = 0
        total = 0
        for images, labels in test_loader:
            out, _ = model1[i](images)
            _, predicted = torch.max(out.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        return(1 - correct / total)


def train2(i, epoch, mislable_rate, loader):
    '''
    This function trains the logistic regression
    '''

    for j, (images, labels) in enumerate(loader):
        _, out_x = model1[i](images)
        out_s = min(torch.norm(out_x, 4), THRESHOLD) / \
            torch.norm(out_x, 4) * out_x
        out = model2[i](out_s)
        loss = weighted_nll(out, labels, mislable_rate)

        optimizer2[i].zero_grad()
        loss.backward()
        optimizer2[i].step()

        total = labels.size(0)
        _, predicted = torch.max(out.data, 1)
        correct = (predicted == labels).sum().item()
    print('LoR: Epoch [{}/{}], Loss: {:.4f}, Training Accuracy: {:.2f}%'.format(
        epoch + 1, NUM_EPOCHS_LOG, loss.item(), (correct / total) * 100))


def test2(i):
    '''
    This function tests the logistic regrssion
    '''

    model1[i].eval()
    model2[i].eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            _, out_x = model1[i](images)
            out_s = min(torch.norm(out_x, 4), THRESHOLD) / \
                torch.norm(out_x, 4) * out_x
            out = model2[i](out_s)
            _, predicted = torch.max(out.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        return 1 - correct / total


# Initiation
model1 = [None] * len(ILIST)
optimizer1 = [None] * len(ILIST)
model2 = [None] * len(ILIST)
optimizer2 = [None] * len(ILIST)

for i in ILIST:
    model1[i] = ConvNet()
    model2[i] = LogisticRegression()
    optimizer1[i] = torch.optim.Adam(model1[i].parameters(), lr=LEARNING_RATE)
    optimizer2[i] = torch.optim.SGD(model2[i].parameters(), lr=0.01)

# Load data
traindata = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            transforms.Normalize(
                (0.1307,
                 ),
                (0.3081,
                 ))]))
idx1 = traindata.train_labels == 4
idx1 += traindata.train_labels == 9
idx_train = np.where(idx1)

dataset_test = torchvision.datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            transforms.Normalize(
                (0.1307,
                 ),
                (0.3081,
                 ))]))
idx2 = dataset_test.test_labels == 4
idx2 += dataset_test.test_labels == 9
idx_test = np.where(idx2)
dataset_test = changeto_01_test(dataset_test)
dataset_test = torch.utils.data.dataset.Subset(
    dataset_test, np.where(idx2 == 1)[0])
test_loader = torch.utils.data.DataLoader(
    dataset=dataset_test,
    batch_size=BATCH_SIZE_TEST,
    shuffle=True)


def multi(i):
    if (i + 1) % 8 == 0:
        print('Processed:{:.2f}%'.format(i + 1))

    dataset_train = torchvision.datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                transforms.Normalize(
                    (0.1307,
                     ),
                    (0.3081,
                     ))]))
    dataset_train = flip_train(
        mislable_rate=MISLABLE_RATE,
        dataset=dataset_train)
    dataset_train = changeto_01_train(dataset_train)
    dataset_train = torch.utils.data.dataset.Subset(
        dataset_train, np.where(idx1 == 1)[0])
    train_loader = torch.utils.data.DataLoader(
        dataset=dataset_train, batch_size=BATCH_SIZE_TRAIN, shuffle=True)

    for epoch in range(NUM_EPOCHS_CNN):
        train1(i, epoch, loader=train_loader, mislable_rate=MISLABLE_RATE)
        # print('Cnn:{}'.format(test1(i)))
    test_acc1 = test1(i)

    for epoch in range(NUM_EPOCHS_LOG):
        train2(i, epoch, loader=train_loader, mislable_rate=MISLABLE_RATE)
        # print('Logostic:{}'.format(test2(i)))
    test_acc2 = test2(i)

    print((test_acc1, test_acc2))
    return test_acc1, test_acc2


pool = multiprocessing.Pool()
acc1, acc2 = zip(*pool.imap(multi, ILIST))
pool.close()
pool.join()

print('Mislabeling Rate: {}, Threshold: {}'.format(MISLABLE_RATE, THRESHOLD))
print(
    'Testing error with original features, Mean: {}, Standard Error: {}'.format(
        mean(acc1),
        stats.sem(acc1)))
print(
    'Testing error with shurnk features, Mean: {}, Standard Error: {}'.format(
        mean(acc2),
        stats.sem(acc2)))
