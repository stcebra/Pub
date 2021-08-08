import os

import torch
from torchvision import transforms

from my_dataset import MyDataSet

from model import DynFilter, DFNet, BMNet
import torch.nn.functional as F

import torch.nn as nn
from torchvision import transforms, datasets, utils
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
import torch.optim as optim
from tqdm import tqdm
from math import ceil
from utils import *
from flow_utils import *
from loss import L1_Charbonnier_loss
from PIL import Image
import skimage.metrics
import torchvision.transforms.functional as TF


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_root = os.path.abspath(os.path.join(os.getcwd(), "./"))  # get data root path
    #root = os.path.join(data_root, "vimeo_interp_test", "vimeo_interp_test", "target", "00001")  #  data set path
    root = os.path.join(data_root, "vimeo_interp_test", "vimeo_interp_test", "target_f")  #  data set path
    #root = os.path.join(data_root, "vimeo_triplet", "sequences")
    assert os.path.exists(root), "{} path does not exist.".format(root)\

    roots = [x for x in os.listdir(root)]
    #print(roots)
    #print(root)
    examples = []
    for y in roots:
        root1 = os.path.join(root,y)
        #print(root1)
        temp = [x for x in os.listdir(root1) if os.path.isdir(os.path.join(root1, x))]
        for z in temp:
            examples.append(os.path.join(root1, z))
    train_images_path1 = []
    train_images_path2 = []
    train_images_target = []

    #print(examples)

    for x_path in examples:
        #print(x_path)

        images = [img for img in os.listdir(x_path)]
        i = 0
        for img in images:
            img_path = os.path.join(x_path, img)
            if i == 0:
                train_images_path1.append(img_path)
            elif i == 2:
                train_images_path2.append(img_path)
            else:
                #print(img_path)
                train_images_target.append(img_path)
            i += 1

    print("Done!")

    data_transform = {
        "train": transforms.Compose([transforms.RandomCrop((256,448)),
                                     transforms.RandomVerticalFlip(),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.RandomRotation(10),
                                     transforms.ToTensor()]),
        "val": transforms.Compose([transforms.ToTensor()])}


    train_data_set = MyDataSet(images_path1=train_images_path1,
                               images_path2=train_images_path2,
                               images_target=train_images_target,
                               transform=data_transform["train"])
    test_data_set = MyDataSet(images_path1=train_images_path1,
                               images_path2=train_images_path2,
                               images_target=train_images_target,
                               transform=data_transform["val"])

    print(len(train_data_set))


    train_loader = torch.utils.data.DataLoader(train_data_set,
                                               batch_size=1,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data_set,
                                               batch_size=1,
                                               shuffle=True)

    net = BMNet().to(device)
    dnet =DFNet(32,4,16,6).to(device)
    ReLU = torch.nn.ReLU()
    ReLU.to(device)
    context_layer = nn.Conv2d(3, 64, (7, 7), stride=(1, 1), padding=(3, 3), bias=False)
    context_layer.load_state_dict(torch.load('Weights/context_layer.pth'))
    context_layer.to(device)
    filtering = DynFilter()
    filtering.to(device)

    print(context_layer)

    net.load_state_dict(torch.load('Weights/BMNet_weights.pth'))
    dnet.load_state_dict(torch.load('Weights/DFNet_weights.pth'))
    for param in net.parameters():
        param.requires_grad = False
    for param in context_layer.parameters():
        param.requires_grad = False
    for param in dnet.parameters():
        param.requires_grad = False

    img1=Image.open(train_images_path1[1]) # /vimeo_interp_test/vimeo_interp_test/target_f/00001/0402/im1
    img2=Image.open(train_images_path2[1]) # /vimeo_interp_test/vimeo_interp_test/target_f/00001/0402/im3
    plt.imshow(img1)
    plt.show()

    print(type(context_layer))

    '''no_of_layers = 0
    conv_layers = []

    model_children = list(context_layer.children())
    for child in model_children:
        if type(child)==nn.Conv2d:
            no_of_layers += 1
            conv_layers.append(child)
        elif type(child)==nn.Sequential:
            for layer in child.children():
                if type(layer)==nn.Conv2d:
                    no_of_layers+=1
                    conv_layers.append(layer)
    print("layer number: {}".format(no_of_layers))'''

    #img = data_transform["val"](img)
    img1, img2 = map(TF.to_tensor, (img1,img2))

    img1 = img1.unsqueeze(0).cuda()
    result = context_layer(img1)
    print(result)

    plt.figure(figsize=(50, 10))
    layer_viz = result[0, :, :, :]
    layer_viz = layer_viz.cpu().data.numpy()
    for i, filter in enumerate(layer_viz):
        if i == 16:
            break
        #print(i,end="")
        #plt.subplot(2, 8, i + 1)
        plt.subplot(4, 4, i + 1)
        plt.imshow(filter, cmap='gray')
        plt.axis("off") # 显示横纵坐标
    plt.show()
    plt.close()

    plt.figure(figsize=(50, 10))
    layer_viz = result[0, :, :, :]
    layer_viz = layer_viz.cpu().data.numpy()
    for i, filter in enumerate(layer_viz):
        if i == 24:
            break
        plt.subplot(3, 8, i + 1)
        plt.imshow(filter, cmap='viridis')
        plt.axis("off")
    plt.show()
    plt.close()





if __name__ == '__main__':
    main()
