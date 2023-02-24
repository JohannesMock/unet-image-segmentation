# The Model is taken from: https://github.com/jarvislabsai/siim-acr/blob/main/Building%20Unet.ipynb

import torch 
from torch import nn
import torch.nn.functional as F
import timm
from fastai.vision.all import Module
from fastai.vision.all import ConvLayer
from fastai.vision.all import noop


class Encoder(Module):
    def __init__(self, model_name='resnext50_32x4d'):
        self.encoder = timm.create_model(model_name, features_only=True, pretrained=True)
        
    def forward(self, x):
        return self.encoder(x)


class UnetBlock(Module):
    def __init__(self,in_channels,chanels,out_channels,attn=None):
        self.conv1 = ConvLayer(in_channels,chanels)
        self.conv2 = ConvLayer(chanels,out_channels)
        self.attn_layer = attn(out_channels) if attn else noop
        
    def forward(self,x):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attn_layer(x)
        return x


def calc_hyperfeats(d1,d2,d3,d4,d5):
    hyperfeats = torch.cat((
                         d1,
                         F.interpolate(d2, scale_factor=2, mode='bilinear',align_corners=False),
                         F.interpolate(d3, scale_factor=4, mode='bilinear', align_corners=False),
                         F.interpolate(d4, scale_factor=8, mode='bilinear', align_corners=False),
                         F.interpolate(d5, scale_factor=16, mode='bilinear', align_corners=False)), 1)
    return hyperfeats


class UnetDecoder(Module):
    def __init__(self, fs=32, expansion=4,n_out=1, hypercol=False,attn=None):
        print(n_out)
        center_ch = 512*expansion
        decoder5_ch = center_ch + (256*expansion)
        channels = 512
        self.hypercol = hypercol
        self.center = nn.Sequential(ConvLayer(center_ch,center_ch),ConvLayer(center_ch,center_ch//2))
        self.decoder5 = UnetBlock(decoder5_ch,channels,fs,attn) 
        self.decoder4 = UnetBlock(256*expansion+fs,256,fs,attn) 
        self.decoder3 = UnetBlock(128*expansion+fs,128,fs,attn)
        self.decoder2 = UnetBlock(64*expansion+fs,64,fs,attn)
        self.decoder1 = UnetBlock(fs,fs,fs,attn)
        if hypercol:
            self.logit = nn.Sequential(ConvLayer(fs*5,fs*2),ConvLayer(fs*2,fs),nn.Conv2d(fs,n_out,kernel_size=1))
        else:
            self.logit = nn.Sequential(ConvLayer(fs,fs//2),ConvLayer(fs//2,fs//2),nn.Conv2d(fs//2,n_out,kernel_size=1))
        
    def forward(self, feats):
        e1,e2,e3,e4,e5 = feats

        f = self.center(e5)
        d5 = self.decoder5(torch.cat([f, e5], 1))
        d4 = self.decoder4(torch.cat([d5, e4], 1))
        d3 = self.decoder3(torch.cat([d4, e3], 1))
        d2 = self.decoder2(torch.cat([d3, e2], 1))
        d1 = self.decoder1(d2)
        return self.logit(calc_hyperfeats(d1,d2,d3,d4,d5)) if self.hypercol else self.logit(d1)
    
        
class Unet(Module):
    def __init__(self, fs=32, expansion=4, model_name='resnext50_32x4d',n_out=1,hypercol=False,attn=None):
        self.encoder = Encoder(model_name)
        self.decoder = UnetDecoder(fs=fs, expansion=expansion,n_out=n_out,hypercol=hypercol,attn=attn)
        
    def forward(self, x):
        feats = self.encoder(x) #'64 256 512 1024 2048'
        out = self.decoder(feats)
        return out