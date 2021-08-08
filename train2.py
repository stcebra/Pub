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
def tv(flow):
    bs_flow, c_flow, h_flow, w_flow = flow.shape
    tv_h = torch.pow(flow[:,:,1:,:] - flow[:,:,:-1,:], 2).sum()
    tv_w = torch.pow(flow[:,:,:,1:] - flow[:,:,:,:-1], 2).sum()
    return (tv_h+tv_w)/(bs_flow*c_flow*h_flow*w_flow)

def photometric_loss(flow, I0, I1, It, weight, loss_function, ReLU, context_layer):
    #print(It.shape)
    H_, W_ = It.shape[-2:]

    #print("weight is: ",end="")
    #print(weight)

    #print("flow shape before unsample: ",end="")
    #print(flow.shape)
    flow = F.interpolate(flow, (H_, W_), mode='bilinear')
    #print("flow shape after unsample: ",end="")
    #print(flow.shape)

    C5 = warp(torch.cat((I0, ReLU(context_layer(I0))), dim=1), flow*(-2*0.5))
    C6 = warp(torch.cat((I1, ReLU(context_layer(I1))), dim=1), flow * 2 * (1-0.5))

    I_t_0 = C5[:,0:3,:,:]
    I_t_1 = C6[:,0:3,:,:]

    #print(weight * (loss_function(I_t_0, It) + loss_function(I_t_1, It)))
    return weight * (loss_function(I_t_0, It) + loss_function(I_t_1, It))

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_root = os.path.abspath(os.path.join(os.getcwd(), "./"))  # get data root path
    #root = os.path.join(data_root, "vimeo_interp_test", "vimeo_interp_test", "target", "00001")  #  data set path
    #root = os.path.join(data_root, "vimeo_interp_test", "vimeo_interp_test", "target")  #  data set path
    root = os.path.join(data_root, "vimeo_triplet", "sequences")
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
    ReLU = torch.nn.ReLU()
    ReLU.to(device)
    context_layer = nn.Conv2d(3, 64, (7, 7), stride=(1, 1), padding=(3, 3), bias=False)
    context_layer.load_state_dict(torch.load('Weights/context_layer.pth'))
    context_layer.to(device)

    i = 0
    '''for param in net.parameters():
        if i == 0:
            continue
        param.requires_grad = False'''
    for param in context_layer.parameters():
        param.requires_grad = False
    epochs = 2500000
    #epochs = 10
    test = 0
    loss_function = L1_Charbonnier_loss()
    lr = 0.0001
    optimizer = optim.Adam(net.parameters(), lr)

    for epoch in range(epochs):
        print("epoch: ",end="")
        print(epoch)

        if epoch // 1000 == 0 and epoch != 0:
            torch.save(dnet.state_dict(), './Weights/{}_bm.pth'.format(epoch))
        # train
        if epoch == 500000 or epoch == 1000000 or epoch == 1500000 or epoch == 2000000:
            lr *= 0.5
            optimizer = optim.Adam(net.parameters(), lr)

        running_loss = 0.0
        train_bar = tqdm(train_loader)
        net.train()
        for step, data in enumerate(train_bar):
            I0, I1, It = data
            I0 = I0.to(device)
            I1 = I1.to(device)
            It = It.to(device)

            flows = net(F.interpolate(torch.cat((I0, I1), dim=1), (256, 448), mode='bilinear'), time=0.5, train=True)

            C5 = warp(torch.cat((I0, ReLU(context_layer(I0))), dim=1), flows[0]*(-2*0.5))
            C6 = warp(torch.cat((I1, ReLU(context_layer(I1))), dim=1), flows[0] * 2 * (1-0.5))

            I_t_0 = C5[:,0:3,:,:]
            I_t_1 = C6[:,0:3,:,:]

            charbonnier_loss = 0
            for i in range(0,len(flows)):
                charbonnier_loss += photometric_loss(flows[i], I0, I1, It, 0.01 * (2 ** (i+1)), loss_function, ReLU, context_layer)


            V_t_0 = flows[0]*(-2*0.5)
            V_t_1 = flows[0] * 2 * (1-0.5)

            smooth_loss1 = tv(V_t_0)
            smooth_loss2 = tv(V_t_1)

            loss = charbonnier_loss + smooth_loss1 + smooth_loss2
            #print(loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    torch.save(net.state_dict(), './Weights/bm.pth')




if __name__ == '__main__':
    main()
