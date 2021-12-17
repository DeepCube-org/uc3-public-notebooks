# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 12:47:21 2021

@author: unknown
"""

import sys 
from captum.attr import Saliency, IntegratedGradients, NoiseTunnel, DeepLift, GuidedGradCam, LayerAttribution, LayerGradCam
import matplotlib.pyplot as plt
import numpy as np 
import torch
from captum.attr import visualization as viz


def attribute_image_features(model, algorithm, sample, labels, ind, **kwargs):
    model.zero_grad()
    tensor_attributions = algorithm.attribute(sample, target=labels[ind], **kwargs)   
    return tensor_attributions


class ActivationMaps():
    """ Class that contains techniques
    to highlight pixels that are maximally activated for a given output """
    
    def __init__(self, params):
        self.params = params
        self.img = self.params["instance"]
        self.method = params["method"]
        self.model = self.params["model"]
    
    def fit(self):    
        if self.method == "gradient":
            self.grad = Saliency(self.model)
            self.attr = self.grad.attribute(self.img, target=int(self.params["y"]))
            
        elif self.method == "integrated_gradients":
            self.grad = IntegratedGradients(self.model)  
            self.attr = self.grad.attribute(self.img, baselines=self.base, target=int(self.params["y"]))
            
        elif self.method == "smoothed_ig":
            self.grad = IntegratedGradients(self.model)
            self.nt = NoiseTunnel(self.grad)
            self.attr = self.nt.attribute(self.img, baselines=self.base, target=int(self.params["y"]), 
                nt_type='smoothgrad_sq', nt_samples=100, stdevs=0.01)
            
        elif self.method == "deeplift":
            self.grad = DeepLift(self.model)
            self.attr = self.grad.attribute(self.img, baselines=self.base, target=int(self.params["y"]))
            
        elif self.method == "gradcam":
            self.grad = LayerGradCam(self.model.forward, self.params["layer"])
            self.attr = self.grad.attribute(self.img, target=int(self.params["y"]))
            print(self.attr.shape)
            self.attr = LayerAttribution.interpolate(self.attr, (self.img.shape[-1],self.img.shape[-1]) )
            
        else:
            sys.exit("Specify a method to evaluate pixel importance!")
                      
        return (self.attr, self.params["value_loss"])

    def plot(self, path, channel = 0, timestep = 0):
        print('Ground Truth:', self.params["classes"][int(self.params["y"])], ' Predicted probability:', 
              float(self.params["pred_proba"]), ' Loss function value: ',  float(self.params["value_loss"]))
        
        print(self.attr.shape)
        if self.params["mode"] == "spatial" and self.method != "gradcam":
            attr_img = np.transpose(self.attr[0,channel,:,:].reshape((1, 25, 25)).cpu().detach().numpy(), (1, 2, 0))
            input_img = self.img[0, channel,:,:]
        elif self.params["mode"] == "spatiotemporal" and self.method != "gradcam":
            if self.params["avg_grad"]:
                avg_attr = torch.mean(self.attr[0,:,:,:,:], 0)
                attr_img = np.transpose(avg_attr[channel,:,:].reshape((1, 25, 25)).cpu().detach().numpy(), (1, 2, 0))
            else:
                attr_img = np.transpose(self.attr[0,timestep,channel,:,:].reshape((1, 25, 25)).cpu().detach().numpy(), (1, 2, 0))
            input_img = self.img[0, timestep, channel,:,:]
        elif self.method == "gradcam":
            print(self.attr.shape)
            input_img = self.img[0, timestep, channel,:,:]
            attr_img = np.transpose(self.attr[0,:,:].reshape((1, 25, 25)).cpu().detach().numpy(), (1, 2, 0))
        else:
            sys.exit("Specify instance type: spatial or spatiotemporal!")
            
        if self.params["plot_feature"]:
#            plt.figure(figsize=(10,10))
            im = plt.imshow(input_img)
            plt.title(self.params["features"][channel])
            plt.colorbar(im)
            plt.show()
        
        if self.params["plot_gradient"]:
#            plt.figure(figsize=(10,10))
            plt.imshow(attr_img)
            plt.imshow(self.params["burned_area"], cmap=plt.cm.cool, interpolation='none', alpha=0.25)
            plt.title("Pixel Activation Map: "+self.params["features"][channel])
#            plt.colorbar(imgr)
            plt.savefig(path)
            plt.show()
        
        if self.params["plot_overlay"]:
            original_img = np.transpose((input_img.reshape((1, 25, 25)).cpu().detach().numpy() / 2) + 0.5, (1, 2, 0))
            _ = viz.visualize_image_attr(attr_img, original_img, method="blended_heat_map", sign="absolute_value",
                           show_colorbar=True, title="Overlayed Activation")


