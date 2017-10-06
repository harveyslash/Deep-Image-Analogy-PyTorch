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


class FeatureExtractor(nn.Module):
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
    def __init__(self):
        self.cnn_temp = models.vgg19(pretrained=True).features
        self.model = FeatureExtractor()  # the new Feature extractor module network
        conv_counter = 1
        relu_counter = 1
        batn_counter = 1

        block_counter = 1

        for layer in list(self.cnn_temp):

            if isinstance(layer, nn.Conv2d):
                name = "conv_" + str(block_counter) + "_" + str(conv_counter)
                conv_counter += 1
                self.model.add_layer(name, layer)

            if isinstance(layer, nn.ReLU):
                name = "relu_" + str(block_counter) + "_" + str(relu_counter)
                relu_counter += 1
                self.model.add_layer(name, nn.ReLU(inplace=False))

            if isinstance(layer, nn.MaxPool2d):
                name = "pool_" + str(block_counter)
                batn_counter = relu_counter = conv_counter = 1
                block_counter += 1
                self.model.add_layer(name, nn.AvgPool2d(2, 2))  # ***

            if isinstance(layer, nn.BatchNorm2d):
                name = "batn_" + str(block_counter) + "_" + str(batn_counter)
                batn_counter += 1
                self.model.add_layer(name, layer)  # ***

    def get_viz_tensor(self, activations_tensor):
        reshaped_tensor = activations_tensor.view(-1, 1, activations_tensor.size()[1], activations_tensor.size()[2])
        grid = make_grid(reshaped_tensor).numpy()
        grid = np.transpose(grid, (1, 2, 0))
        return grid

    def get_features_for_layer(self, img_tensor, layer_num):

        feature = self.model(img_tensor)[layer_num].data
        return feature
