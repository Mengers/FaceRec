'''
@Descripttion: This is menger's demo,which is only for reference
@version:
@Author: menger
@Date: 2020-11-20 09:32:00
@LastEditors:
@LastEditTime:
'''
from itertools import product

import torch
from math import sqrt
# from config import *


image_size = 640
strides = [4,4,8,8,16,32,32,32]
feature_maps = [159,159,79,79,39,19,19,19]
scale_factors = [15, 20, 40, 70, 110, 250, 400, 560]
scales = [(10,15),(15,20),(20,40),(40,70),(70,110),(110,250),(250,400),(400,560)]


class Priors:
    def __init__(self,clip = True):
        self.image_size = image_size
        self.strides = strides
        self.feature_maps = feature_maps
        self.clip = clip
    def __call__(self):
        priors = []
        for k, f in enumerate(self.feature_maps):
            # 513/4 = 128.25
            # 126/128.25 = 0.98245
            # 126.5/128.25 = 0.98635
            scale = self.image_size / self.strides[k]
            for i, j in product(range(f), repeat=2):

                cx = (j + 0.5) / scale
                cy = (i + 0.5) / scale
                scale_factor = scale_factors[k] * 0.5 / self.image_size
                #r = self.sizes[k]
                #r = r/self.image_size
                #h = w = self.sizes[k] / self.image_size
                priors.append([cx, cy, scale_factor])

        priors = torch.tensor(priors) #(num_priors,4)
        if self.clip:
            priors.clamp_(max=1, min=0)
        return priors
def get_priors():
    return Priors()
if __name__ == "__main__":
    priors = Priors()
    print(priors().size())
    print(priors().data)
