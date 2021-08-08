import os

import torch
from torchvision import transforms

from my_dataset import MyDataSet

from model import DynFilter, DFNet, BMNet

import torch.nn as nn
from torchvision import transforms, datasets, utils
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from tqdm import tqdm
from math import ceil
from utils import *
from flow_utils import *

'''def l1_loss(predictions, targets):
  """Implements tensorflow l1 loss.
  Args:
  Returns:
  """
  total_elements = (tf.shape(targets)[0] * tf.shape(targets)[1] * tf.shape(targets)[2]
      * tf.shape(targets)[3])
  total_elements = tf.to_float(total_elements)

  loss = tf.reduce_sum(tf.abs(predictions- targets))
  loss = tf.div(loss, total_elements)
  return loss'''

class L1_Charbonnier_loss(nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-6

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt( diff * diff + self.eps * self.eps )
        loss = torch.sum(error)
        return loss

ReLU = torch.nn.ReLU()
torch.cuda.empty_cache()

data_root = os.path.abspath(os.path.join(os.getcwd(), "./"))  # get data root path
root = os.path.join(data_root, "vimeo_interp_test", "vimeo_interp_test", "target", "00001")  #  data set path
assert os.path.exists(root), "{} path does not exist.".format(root)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    print("using {} device.".format(device))

    #train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(root)

    examples = [x for x in os.listdir(root) if os.path.isdir(os.path.join(root, x))]
    train_images_path1 = []
    train_images_path2 = []
    train_images_target = []

    for x in examples:
        x_path = os.path.join(root, x)
        #print(x_path)

        images = [img for img in os.listdir(x_path)]
        i = 0
        for img in images:
            img_path = os.path.join(x_path, img)
            #print(img_path)
            if i == 0:
                train_images_path1.append(img_path)
            elif i == 2:
                train_images_path2.append(img_path)
            else:
                train_images_target.append(img_path)
            i += 1

    print("Done!")



    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(256),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    train_data_set = MyDataSet(images_path1=train_images_path1,
                               images_path2=train_images_path2,
                               images_target=train_images_target,
                               transform=data_transform["train"])

    print(len(train_data_set))


    train_loader = torch.utils.data.DataLoader(train_data_set,
                                               batch_size=1,
                                               shuffle=True)


    net = BMNet()
    #dfnet = DFNet(32,4,16,6)
    #dfnet.to(device)
    cnet = nn.Conv2d(3, 64, (7, 7), stride=(1, 1), padding=(3, 3), bias=False)
    #cnet.to(device)
    cnet.load_state_dict(torch.load('Weights/context_layer.pth'))
    net.to(device)
    # change this with bilateral loss function
    #loss_function = nn.MSELoss()
    loss_function = L1_Charbonnier_loss()
    # pata = list(net.parameters())
    optimizer = optim.Adam(net.parameters(), lr=0.0001)

    epochs = 10
    best_acc = 0.0
    train_steps = len(train_loader)
    #df = DynFilter()
    #df.to(device)
    #net.train()
    net.load_state_dict(torch.load('Weights/BMNet_weights.pth'))
    for epoch in range(epochs):
        # train

        running_loss = 0.0
        train_bar = tqdm(train_loader)
        for step, data in enumerate(train_bar):
            images_path1, images_path2, images_target = data
            input = torch.cat((images_path1, images_path2), dim=1)
            optimizer.zero_grad()
            F01 = net(input.to(device), time=0) * 2.0
            F10 = net(input.to(device), time=1) * -2.0
            # 我把BM这行comment掉是能跑起来的，加上就GPU不够了
            BM = net(input.to(device), time=0.5)
            #torch.cuda.empty_cache()

            #C1 = warp(torch.cat((images_path1, ReLU(cnet(images_path1))), dim=1).to(device), (-0.5) * F01)   # F_t_0
            #C2 = warp(torch.cat((images_path2, ReLU(cnet(images_path2))), dim=1).to(device), (1-0.5) * F01)  # F_t_1
            #C3 = warp(torch.cat((images_path1, ReLU(cnet(images_path1))), dim=1).to(device), (0.5) * F10)  # F_t_0
            #C4 = warp(torch.cat((images_path2, ReLU(cnet(images_path2))), dim=1).to(device), (0.5) * F10)   # F_t_1
            #C5 = warp(torch.cat((images_path1, ReLU(cnet(images_path1))), dim=1).to(device), BM*(-2*0.5))
            #C6 = warp(torch.cat((images_path2, ReLU(cnet(images_path2))), dim=1).to(device), BM * 2 * (1-0.5))



            '''input1 = torch.cat((images_path1.to(device),C1.to(device),C2.to(device),C3.to(device),C4.to(device),C5.to(device),C6.to(device),images_path2.to(device)),dim=1).to(device)
            DF = torch.nn.functional.softmax(dfnet(input1),dim=1)

            candidates = input1[:,3:-3,:,:]

            R = df(candidates[:, 0::67, :, :], DF)
            G = df(candidates[:, 1::67, :, :], DF)
            B = df(candidates[:, 2::67, :, :], DF)

            I2 = torch.cat((R, G, B), dim=1)'''

            #print(F01.shape)
            #print(torch.cat((F01,F10), dim=1).shape)
            #print(images_target[:,:4,:,:].shape)
            # change this with bilateral loss function
            #loss = loss_function(BM.to(device), images_target[:,:2,:,:].to(device))
            #loss.backward()
            #optimizer.step()

            '''loss = loss_function(I2.to(device), images_target.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()'''


            # print statistics
            #running_loss += loss.item()

            #train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,epochs,loss)

    torch.save(dfnet.state_dict(), './Weights/b.pth')
    torch.save(df.state_dict(), './Weights/c.pth')

    for parameters in net.parameters():
        print(parameters)
    torch.save(net.state_dict(), './Weights/a.pth')
    print('Finished Training')


if __name__ == '__main__':
    main()
