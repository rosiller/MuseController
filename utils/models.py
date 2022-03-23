import numpy as np
import torch as t
import torch.nn.functional as F

class Flatten(t.nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class EEGNet(t.nn.Module):
    def __init__(self,nb_classes = 5):
        super(EEGNet, self).__init__()
        self.T = 256  # Timepoints
        self.C = 4    # Channels
        self.N = nb_classes   # Classes
        
        self.F1 = 8
        self.D = 2
        self.F2 = self.F1 * self.D
        self.kern_length = 64
        self.p_dropout = 0.2
        self.n_features = (self.T // 8) // 8

        # Block 1 
        self.conv1_pad = t.nn.ZeroPad2d((63, 64, 0, 0))
        self.conv1 = t.nn.Conv2d(1, self.F1, (1, 128), bias=False)
        self.batch_norm1 = t.nn.BatchNorm2d(self.F1, momentum=0.01, eps=0.001) 
        self.conv2 = t.nn.Conv2d(self.F1, self.F2, (self.C, 1), groups=self.F1, bias=False) # Depthwise convolution
        self.batch_norm2 = t.nn.BatchNorm2d(self.F2, momentum=0.01, eps=0.001)
        self.activation1 = t.nn.ELU(inplace=True)
        self.pool1 = t.nn.AvgPool2d((1, 8)) 
        self.dropout1 = t.nn.Dropout(p=self.p_dropout)

        # Block 2
        self.sep_conv_pad = t.nn.ZeroPad2d((7, 8, 0, 0))
        self.sep_conv1 = t.nn.Conv2d(self.F2, self.F2, (1, 16), groups=self.F2, bias=False)
        self.sep_conv2 = t.nn.Conv2d(self.F2, self.F2, (1, 1), bias=False)
        self.batch_norm3 = t.nn.BatchNorm2d(self.F2, momentum=0.01, eps=0.001) 
        self.activation2 = t.nn.ELU(inplace=True) 
        self.pool2 = t.nn.AvgPool2d((1, 8))
        self.dropout2 = t.nn.Dropout(p=self.p_dropout)

        # Classification
        self.flatten = Flatten()
        self.fc = t.nn.Linear(self.F2 * self.n_features, self.N, bias=True)
             

    def forward(self, x):
        # Block 1
        x = self.conv1_pad(x)
        x = self.conv1(x)   
        x = self.batch_norm1(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.activation1(x)
        x = self.pool1(x)            
        x = self.dropout1(x)

        # Block2
        x = self.sep_conv_pad(x)
        x = self.sep_conv1(x)        
        x = self.sep_conv2(x)        
        x = self.batch_norm3(x)
        x = self.activation2(x)
        x = self.pool2(x)            
        x = self.dropout2(x)

        # Classification
        x = self.flatten(x)          
        x = self.fc(x)               

        return x