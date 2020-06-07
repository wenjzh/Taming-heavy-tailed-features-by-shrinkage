# This script implements numerical study on MNIST dataset. The main idea is to compare 
# the classification performance from original CNN and CNN with a l4-norm shrinkage layer.

# Arthor: Wenjing Zhou, Ziwei Zhu
# Last modified date: 06/07/2020

import warnings
from scipy.stats import kurtosis
import multiprocessing
from scipy import stats
from skimage.util import random_noise
from torchvision.utils import save_image
from statistics import mean
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch.nn as nn
import torchvision
import torch
import numpy as np
import random
import math
MISLABEL_RATE = 0.4
NOISE_LEVEL = 0.6
NOISE_TYPE = 'salt'
THRESHOLD = 2
ILIST = range(100)
NUM_EPOCHS = 13
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


class RandomNoise(object):
    '''
    This function adds additional noise to the tensors from images
    @input tensor: the original tensor
    @param mode: the noise type, e.g. 'salt'
    @param amount: the propotion of image pixels to replace with noise
    @return: the tensor with noise
    '''

    def __call__(self, tensor):
        return torch.tensor(
            random_noise(
                tensor,
                mode=NOISE_TYPE,
                amount=NOISE_LEVEL,
                clip=True))


class ConvNet(nn.Module):

    '''Original CNN'''

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
        out_x = self.fc1(out)
        out = self.fc2(out_x)
        return out, out_x


class ConvNetS(nn.Module):

    ''' Shrinkage CNN'''

    def __init__(self):
        super(ConvNetS, self).__init__()
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
        out_x = self.fc1(out)

        # shrink
        out_s = min(torch.norm(out_x, 4), THRESHOLD) / \
            torch.norm(out_x, 4) * out_x

        out = self.fc2(out_s)
        return out


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


def train(i, epoch, mislable_rate, loader):
    '''
    This function trains the original CNN
    '''

    for _, (images, labels) in enumerate(loader):
        out, _ = model[i](images)  # ConvNet: out, out_x
        loss = weighted_nll(out, labels, mislable_rate)

        optimizer[i].zero_grad()
        loss.backward()
        optimizer[i].step()

        total = labels.size(0)
        _, predicted = torch.max(out.data, 1)
        correct = (predicted == labels).sum().item()
    print('CNN Epoch [{}/{}], Loss: {:.4f}, Training Accuracy: {:.2f}%'.format(
        epoch + 1, NUM_EPOCHS, loss.item(), (correct / total) * 100))


def test(i):
    '''
    This function tests the original CNN
    '''

    model[i].eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            outputs, _ = model[i](images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        return(1 - correct / total)


def trainS(i, epoch, mislable_rate, loader):
    '''
    This function trains the shrinkage CNN
    '''

    for _, (images, labels) in enumerate(loader):
        out = modelS[i](images)  # out
        loss = weighted_nll(out, labels, mislable_rate)

        optimizerS[i].zero_grad()
        loss.backward()
        optimizerS[i].step()

        total = labels.size(0)
        _, predicted = torch.max(out.data, 1)
        correct = (predicted == labels).sum().item()
    print('Shrunk: Epoch [{}/{}], Loss: {:.4f}, Training Accuracy: {:.2f}%'.format(
        epoch + 1, NUM_EPOCHS, loss.item(), (correct / total) * 100))


def testS(i):
    '''
    This function tests the shrinkage CNN
    '''

    modelS[i].eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            out = modelS[i](images)
            _, predicted = torch.max(out.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        return(1 - correct / total)


# Load the training dataset
dataset_train = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.1307,
                 ),
                (0.3081,
                 )),
            RandomNoise()]))
idx1 = dataset_train.train_labels == 4
idx1 += dataset_train.train_labels == 9
idx_train = np.where(idx1)
dataset_train = torch.utils.data.dataset.Subset(
    dataset_train, np.where(idx1 == 1)[0])
train_loader_0 = torch.utils.data.DataLoader(
    dataset=dataset_train,
    batch_size=BATCH_SIZE_TRAIN,
    shuffle=True)

# View the noisy images
for data in train_loader_0:
    img = data[0]
    img = img.view(img.size(0), 1, 28, 28)
    save_image(img, '{}_{}.png'.format(NOISE_TYPE, NOISE_LEVEL))

# Load the testing dataset
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

# Initiate the models and optimizers
sm = nn.Softmax()
nloss = nn.NLLLoss()

model = [None] * len(ILIST)
optimizer = [None] * len(ILIST)
modelS = [None] * len(ILIST)
optimizerS = [None] * len(ILIST)

for i in ILIST:
    model[i] = ConvNet()
    modelS[i] = ConvNetS()
    optimizer[i] = torch.optim.Adam(model[i].parameters(), lr=LEARNING_RATE)
    optimizerS[i] = torch.optim.Adam(modelS[i].parameters(), lr=LEARNING_RATE)


def multi(i):
    '''
    This function trains and tests the original CNN and shrinkage CNN
    @input i: the epoch id
    @return test1, test2: the testing classification error based on original, shrinkage CNN
    '''

    if (i + 1) % 8 == 0:
        print('Processed:{:.2f}%'.format(i + 1))

    dataset_train = torchvision.datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.1307,
                     ),
                    (0.3081,
                     )),
                RandomNoise()]))
    dataset_train = flip_train(
        mislable_rate=MISLABEL_RATE,
        dataset=dataset_train)
    dataset_train = changeto_01_train(dataset_train)
    dataset_train = torch.utils.data.dataset.Subset(
        dataset_train, np.where(idx1 == 1)[0])
    train_loader = torch.utils.data.DataLoader(
        dataset=dataset_train, batch_size=BATCH_SIZE_TRAIN, shuffle=True)

    for epoch in range(NUM_EPOCHS):
        train(i, epoch, loader=train_loader, mislable_rate=MISLABEL_RATE)
        trainS(i, epoch, loader=train_loader, mislable_rate=MISLABEL_RATE)
        # print('CNN Epoch [{}/{}], Test Accuracy: {:.2f}%'.format(epoch + 1, num_epochs, test(i)*100))
        # print('SNN Epoch [{}/{}], Test Accuracy: {:.2f}%'.format(epoch + 1, num_epochs, testS(i)*100))
    test1 = test(i)
    test2 = testS(i)
    print((test1, test2))
    return test1, test2


# Parallel Computing
# err1, err2 are tuples containing testing misclassification errors
pool = multiprocessing.Pool()
err1, err2 = zip(*pool.imap(multi, ILIST))
pool.close()
pool.join()

print(
    'Mislabeling Rate: {}, Noise Level: {}, Noise Level: {} '.format(
        MISLABEL_RATE,
        NOISE_TYPE,
        NOISE_LEVEL))
print('Threshold for Shrinkage: {} '.format(THRESHOLD))
print(
    'Testing error with original features, Mean: {}, Standard Error: {}'.format(
        mean(err1),
        stats.sem(err1)))
print(
    'Testing error with shurnk CNN, Mean: {}, Standard Error: {}'.format(
        mean(err2),
        stats.sem(err2)))
