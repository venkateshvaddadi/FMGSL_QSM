#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 17:25:46 2020

@author: cds
"""


import torch
import torch.nn as nn

class QSMnet(nn.Module):
    
    def __init__(self):
        super(QSMnet, self).__init__()
        
        # (inp channels, Out Channels, Kernel Size, Stride, Padding)
        
        # Down 1
        
        self.conv10 = nn.Sequential(nn.Conv3d(1,32,5,1,2),nn.BatchNorm3d(32),
        nn.LeakyReLU(0.1))
        
        self.conv11 = nn.Sequential(nn.Conv3d(32,32,5,1,2),nn.BatchNorm3d(32),
        nn.LeakyReLU(0.1))
        
        self.down1  = nn.MaxPool3d(2,2)
        
        # Down 2
        
        self.conv20 = nn.Sequential(nn.Conv3d(32,64,5,1,2),nn.BatchNorm3d(64),
        nn.LeakyReLU(0.1))
        
        self.conv21 = nn.Sequential(nn.Conv3d(64,64,5,1,2),nn.BatchNorm3d(64),
        nn.LeakyReLU(0.1))
        
        self.down2  = nn.MaxPool3d(2,2)

        # Down 3
        
        self.conv30 = nn.Sequential(nn.Conv3d(64,128,5,1,2),nn.BatchNorm3d(128),
        nn.LeakyReLU(0.1))
        
        self.conv31 = nn.Sequential(nn.Conv3d(128,128,5,1,2),nn.BatchNorm3d(128),
        nn.LeakyReLU(0.1))
        
        self.down3  = nn.MaxPool3d(2,2)

         # Down 4
        
        self.conv40 = nn.Sequential(nn.Conv3d(128,256,5,1,2),nn.BatchNorm3d(256),
        nn.LeakyReLU(0.1))
        
        self.conv41 = nn.Sequential(nn.Conv3d(256,256,5,1,2),nn.BatchNorm3d(256),
        nn.LeakyReLU(0.1))
        
        self.down4  = nn.MaxPool3d(2,2)

        # Middle

        self.convm1 = nn.Sequential(nn.Conv3d(256,512,5,1,2),nn.BatchNorm3d(512),
        nn.LeakyReLU(0.1))
        
        self.convm2 = nn.Sequential(nn.Conv3d(512,512,5,1,2),nn.BatchNorm3d(512),
        nn.LeakyReLU(0.1))

        # Up 1

        self.up1   = nn.ConvTranspose3d(512,256,2,2)

        self.conv50 = nn.Sequential(nn.Conv3d(512,256,5,1,2),nn.BatchNorm3d(256),
        nn.LeakyReLU(0.1))
        
        self.conv51 = nn.Sequential(nn.Conv3d(256,256,5,1,2),nn.BatchNorm3d(256),
        nn.LeakyReLU(0.1))

        # Up 2

        self.up2   = nn.ConvTranspose3d(256,128,2,2)

        self.conv60 = nn.Sequential(nn.Conv3d(256,128,5,1,2),nn.BatchNorm3d(128),
        nn.LeakyReLU(0.1))
        
        self.conv61 = nn.Sequential(nn.Conv3d(128,128,5,1,2),nn.BatchNorm3d(128),
        nn.LeakyReLU(0.1))

        # Up 3

        self.up3   = nn.ConvTranspose3d(128,64,2,2)

        self.conv70 = nn.Sequential(nn.Conv3d(128,64,5,1,2),nn.BatchNorm3d(64),
        nn.LeakyReLU(0.1))
        
        self.conv71 = nn.Sequential(nn.Conv3d(64,64,5,1,2),nn.BatchNorm3d(64),
        nn.LeakyReLU(0.1))

        # Up 4

        self.up4   = nn.ConvTranspose3d(64,32,2,2)

        self.conv80 = nn.Sequential(nn.Conv3d(64,32,5,1,2),nn.BatchNorm3d(32),
        nn.LeakyReLU(0.1))
        
        self.conv81 = nn.Sequential(nn.Conv3d(32,32,5,1,2),nn.BatchNorm3d(32),
        nn.LeakyReLU(0.1))

        # Final

        self.out = nn.Conv3d(32,1,1,1)


        
    def forward(self,x):

      # Down
        
        x10 = self.conv10(x)
        x11 = self.conv11(x10)
        
        xd1 = self.down1(x11)
        
        x20 = self.conv20(xd1)
        x21 = self.conv21(x20)
        
        xd2 = self.down2(x21)

        x30 = self.conv30(xd2)
        x31 = self.conv31(x30)
        
        xd3 = self.down3(x31)

        x40 = self.conv40(xd3)
        x41 = self.conv41(x40)
        
        xd4 = self.down4(x41)

      # Middle

        xm1 = self.convm1(xd4)
        xm2 = self.convm2(xm1)

      # Up

        xu1  = self.up1(xm2)      

        xc1  = torch.cat((xu1,x41),dim = 1)

        x50 = self.conv50(xc1)
        x51 = self.conv51(x50)

        xu2  = self.up2(x51)
      
        xc2  = torch.cat((xu2,x31),dim = 1)

        x60 = self.conv60(xc2)
        x61 = self.conv61(x60)

        xu3  = self.up3(x61)
      
        xc3  = torch.cat((xu3,x21),dim = 1)

        x70 = self.conv70(xc3)
        x71 = self.conv71(x70)

        xu4  = self.up4(x71)
      
        xc4  = torch.cat((xu4,x11),dim = 1)

        x80 = self.conv80(xc4)
        x81 = self.conv81(x80)

        Out = self.out(x81)        
        
        return Out

    def getnumberofparams(self,net):
      pp=0
      for p in (net.parameters()):
        nn=1
        for s in p.size():
          nn=nn*s
        pp+=nn      
      return pp


if __name__ == "__main__":
    net=QSMnet()
    net = net.cuda()
    xx = torch.rand(1,1,64,64,64).cuda()
    yy = net(xx)
    print("Input Shape : ", xx.shape)
    print("Output Shape: ", yy.shape)
    print("Params : ", net.getnumberofparams(net))