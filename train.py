import os

import torch
from torchvision import transforms

from my_dataset import MyDataSet

from model import DynFilter, DFNet, BMNet
import torch.nn.functional as F

import torch.nn as nn
from torchvision import transforms, datasets, utils
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from tqdm import tqdm
from math import ceil
from utils import *
from flow_utils import *
from loss import L1_Charbonnier_loss
from PIL import Image


# [1, 2, 256, 448]
'''def tv(flow):
    bs_flow, c_flow, h_flow, w_flow = flow.shape
    tv_h = torch.pow(flow[:,:,1:,:] - flow[:,:,:-1,:], 2).sum()
    tv_w = torch.pow(flow[:,:,:,1:] - flow[:,:,:,:-1], 2).sum()
    return (tv_h+tv_w)/(bs_flow*c_flow*h_flow*w_flow)'''


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    print("using {} gpus.".format(torch.cuda.device_count()))
    print("using {} device name.".format(torch.cuda.get_device_name(0)))
    print("using {} current device.".format(torch.cuda.current_device()))

    data_root = os.path.abspath(os.path.join(os.getcwd(), "./"))  # get data root path
    #root = os.path.join(data_root, "vimeo_interp_test", "vimeo_interp_test", "target", "00001")  #  data set path
    root = os.path.join(data_root, "vimeo_interp_test", "vimeo_interp_test", "target")  #  data set path
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

    print(len(train_data_set))


    train_loader = torch.utils.data.DataLoader(train_data_set,
                                               batch_size=4,
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

    net.load_state_dict(torch.load('Weights/BMNet_weights.pth'))
    for param in net.parameters():
        param.requires_grad = False
    for param in context_layer.parameters():
        param.requires_grad = False
    #epochs = 1250000
    epochs = 1
    test = 0
    loss_function = L1_Charbonnier_loss()
    lr = 0.0001
    optimizer = optim.Adam(net.parameters(), lr)
    k = 0

    for epoch in range(epochs):
        #print("epoch: ",end="")
        #print(epoch)
        # train
        if epoch // 1000 == 0 and epoch != 0:
            torch.save(dnet.state_dict(), './Weights/{}_def.pth'.format(epoch))
        if epoch == 500000 or epoch == 1000000 or epoch == 750000:
            lr *= 0.5
            optimizer = optim.Adam(net.parameters(), lr)

        running_loss = 0.0
        train_bar = tqdm(train_loader)
        dnet.train()
        for step, data in enumerate(train_bar):
            I0, I1, It = data
            I0 = I0.to(device)
            I1 = I1.to(device)
            It = It.to(device)

            F_0_1 = net(F.interpolate(torch.cat((I0, I1), dim=1), (256, 448), mode='bilinear'), time=0) * 2.0
            F_1_0 = net(F.interpolate(torch.cat((I0, I1), dim=1), (256, 448), mode='bilinear'), time=1) * (-2.0)
            BM = net(F.interpolate(torch.cat((I0, I1), dim=1), (256, 448), mode='bilinear'), time=0.5)

            C1 = warp(torch.cat((I0, ReLU(context_layer(I0))), dim=1), (-0.5) * F_0_1)   # F_t_0
            C2 = warp(torch.cat((I1, ReLU(context_layer(I1))), dim=1), (1-0.5) * F_0_1)  # F_t_1
            C3 = warp(torch.cat((I0, ReLU(context_layer(I0))), dim=1), (0.5) * F_1_0)  # F_t_0
            C4 = warp(torch.cat((I1, ReLU(context_layer(I1))), dim=1), (0.5-1) * F_1_0)   # F_t_1
            C5 = warp(torch.cat((I0, ReLU(context_layer(I0))), dim=1), BM*(-2*0.5))
            C6 = warp(torch.cat((I1, ReLU(context_layer(I1))), dim=1), BM * 2 * (1-0.5))

            #print(tv(BM*(-2*0.5)))

            input = torch.cat((I0,C1,C2,C3,C4,C5,C6,I1),dim=1)
            DF = F.softmax(dnet(input),dim=1)

            candidates = input[:,3:-3,:,:]

            R = filtering(candidates[:, 0::67, :, :], DF)
            G = filtering(candidates[:, 1::67, :, :], DF)
            B = filtering(candidates[:, 2::67, :, :], DF)

            I2 = torch.cat((R, G, B), dim=1)

            loss = loss_function(I2, It)
            #print(loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            torch.save(dnet.state_dict(), './Weights/{}_def.pth'.format(k))
            k+=1

    torch.save(dnet.state_dict(), './Weights/def.pth')




if __name__ == '__main__':
    print("Hello!")
    main()
