import time
import os
from src.PatchMatch import PatchMatchOrig
import torchvision.models as models
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import torch.utils.model_zoo as model_zoo
import cv2
import torchvision
from torchvision import transforms
from torchvision.utils import make_grid
from PIL import Image
from collections import OrderedDict
from PIL import Image
# import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau


class FeatureExtractor(nn.Sequential):
    def __init__(self):
        super(FeatureExtractor, self).__init__()

    def add_layer(self, name, layer):
        self.add_module(name, layer)

    def forward(self, x):
        list = []
        for module in self._modules:
            x = self._modules[module](x)
            list.append(x)
        return list


class VGG19:
    def __init__(self,use_cuda=True):
        self.cnn_temp = models.vgg19(pretrained=True).features
        self.model = FeatureExtractor()  # the new Feature extractor module network
        conv_counter = 1
        relu_counter = 1
        batn_counter = 1

        block_counter = 1
        self.use_cuda = use_cuda

        for i, layer in enumerate(list(self.cnn_temp)):

            if isinstance(layer, nn.Conv2d):
                name = "conv_" + str(block_counter) + "_" + str(conv_counter) + "__" + str(i)
                conv_counter += 1
                self.model.add_layer(name, layer)

            if isinstance(layer, nn.ReLU):
                name = "relu_" + str(block_counter) + "_" + str(relu_counter) + "__" + str(i)
                relu_counter += 1
                self.model.add_layer(name, nn.ReLU(inplace=False))

            if isinstance(layer, nn.MaxPool2d):
                name = "pool_" + str(block_counter) + "__" + str(i)
                batn_counter = relu_counter = conv_counter = 1
                block_counter += 1
                self.model.add_layer(name, nn.AvgPool2d((2,2)))  # ***


            if isinstance(layer, nn.BatchNorm2d):
                name = "batn_" + str(block_counter) + "_" + str(batn_counter) + "__" + str(i)
                batn_counter += 1
                self.model.add_layer(name, layer)  # ***

        if use_cuda:
            self.model.cuda()
            

    def forward_subnet(self, input_tensor, start_layer, end_layer):
        for i, layer in enumerate(list(self.model)):
            if i >= start_layer and i <= end_layer:
                input_tensor = layer(input_tensor)
        return input_tensor

    def get_features(self, img_tensor):
        if self.use_cuda:
            img_tensor = img_tensor.cuda()

        features = self.model(img_tensor)
        features = [i.data.squeeze().cpu().numpy().transpose(1,2,0) for i in features]
        return np.array(features)

    def get_deconvoluted_feat(self,feat,feat_layer_num,iters=13):
        
        def cn_last(th_array):
            return th_array.transpose(1,2,0)

        def cn_first(th_array):
            return th_array.transpose(2,0,1)

        feat = cn_first(feat)
        feat = torch.from_numpy(feat).float()
        scale = 1

        if feat_layer_num == 5:
            start_layer,end_layer = 21,29
            noise = np.random.uniform(size=(1,512,28*scale,28*scale),low=0 , high=1) 
        elif feat_layer_num == 4:
            start_layer,end_layer = 12,20
            noise = np.random.uniform(size=(1,256,56*scale,56*scale),low=0 , high=1) 
        elif feat_layer_num == 3:
            start_layer,end_layer = 7,11
            noise = np.random.uniform(size=(1,128,112*scale,112*scale),low=0 , high=1) 
        elif feat_layer_num == 2:
            start_layer,end_layer = 2,6
            noise = np.random.uniform(size=(1,64,224*scale,224*scale),low=0 , high=1) 
        else:
            print("Invalid layer number")
        # noise = Variable(torch.from_numpy(noise).float()) # use this if you want custom noise 
        noise = torch.randn(noise.shape).float()

        
        if self.use_cuda:
            noise = noise.cuda()
            feat = feat.cuda()

        noise = Variable(noise,requires_grad=True)
        feat = Variable(feat)
        optimizer = optim.Adam([noise],lr=1)

        loss_hist = []
        for i in range(1,iters):
            optimizer.zero_grad()
            output = self.forward_subnet(input_tensor=noise,start_layer=start_layer,end_layer=end_layer)

            diff = output - feat
            norm = torch.norm(diff,p=2)
            loss_value = norm**2

            loss_value.backward()
            optimizer.step()
            noise.data.clamp_(0., 1.)

            loss_hist.append(loss_value.cpu().data.numpy())

        # plt.plot(loss_hist)
        # plt.show()


        noise.data.clamp_(0., 1.)
        noise_cpu = noise.cpu().data.squeeze().numpy()
        del feat 
        del noise
        return cn_last(noise_cpu)
