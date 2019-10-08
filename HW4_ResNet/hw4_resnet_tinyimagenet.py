# -*- coding: utf-8 -*-
"""HW4_resnet_tinyimagenet.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1OO8ZII3lOpODVeDhJLCzqEUnOqhy_-vV
"""

import torch
import torch.nn as nn
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import time
import pandas as pd
import torchvision.datasets as datasets
import os

# For Colab, mount the working directory to Google Drive
"""
from google.colab import drive
import os
drive.mount('/content/drive/')
os.chdir("drive/My Drive/DeepLearningHw/HW4")
"""

# Import self-defined model and utilities during model training
import MyResNet
import MyUtils

# hyperparameters
num_epochs = 50
batch_size = 128
learning_rate = 0.001

transform_train = transforms.Compose([
    transforms.RandomCrop(64, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_val = transforms.Compose([
    transforms.ToTensor(),
])

def create_val_folder(val_dir):
    """
    This method is responsible for separating validation images into separate sub folders
    """
    path = os.path.join(val_dir, 'images')  # path where validation data is present now
    filename = os.path.join(val_dir, 'val_annotations.txt')  # file where image2class mapping is present
    fp = open(filename, "r")  # open file in read mode
    data = fp.readlines()  # read line by line

    # Create a dictionary with image names as key and corresponding classes as values
    val_img_dict = {}
    for line in data:
        words = line.split("\t")
        val_img_dict[words[0]] = words[1]
    fp.close()
    # Create folder if not present, and move image into proper folder
    for img, folder in val_img_dict.items():
        newpath = (os.path.join(path, folder))
        if not os.path.exists(newpath):  # check if folder exists
            os.makedirs(newpath)
        if os.path.exists(os.path.join(path, img)):  # Check if image exists in default directory
            os.rename(os.path.join(path, img), os.path.join(newpath, img))
    return

# Your own directory to the train folder of tiyimagenet
train_dir = '/u/training/tra193/HW_haoren/tiny-imagenet-200/train/'
train_dataset = datasets.ImageFolder(train_dir, transform=transform_train)


# To check the index for each classes
print(train_dataset.class_to_idx)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, 
                                           shuffle=True, num_workers=8)
# Your own directory to the validation folder of tiyimagenet
val_dir = '/u/training/tra193/HW_haoren/tiny-imagenet-200/val/'


if 'val_' in os.listdir(val_dir+'images/')[0]:
    create_val_folder(val_dir)
    val_dir = val_dir+'images/'
else:
    val_dir = val_dir+'images/'


val_dataset = datasets.ImageFolder(val_dir, transform=transforms.ToTensor())
# To check the index for each classes
print(val_dataset.class_to_idx)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, 
                                         shuffle=False, num_workers=8)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

model = MyResNet.MyResNet(MyResNet.BasicBlock, [2, 4, 4, 2], 100, 3).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

total_step = len(trainloader)

acc_df = pd.DataFrame(columns=["Train Accuracy", "Test Accuracy"])
for epoch in range(num_epochs):
    # Train the model
    train_acc, test_acc = MyUtils.train(epoch, model, train_loader, val_loader,
                                        device, optimizer, criterion, num_epochs,
                                        total_step)

    acc_df = acc_df.append({"Train Accuracy": train_acc,
                            "Test Accuracy": test_acc},
                           ignore_index=True)

print("\nThe accuracy on the test set is: {:.2} %"
        .format(calculate_accuracy(model, testloader)))

# save the accuracy
acc_df.to_csv("./accuracy_tinyimagenet.csv")