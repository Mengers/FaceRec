'''
@Descripttion: This is menger's demo,which is only for reference
@version:
@Author: menger
@Date: 2020-12-12
@LastEditors:
@LastEditTime:
'''

import sys
sys.path.append("..")

from torch.utils.data import Dataset
import torch.nn.functional as F


import glob
from PIL import Image
import xml.etree.ElementTree as ET
import numpy as np
from lib.priors import Priors
from lib.utils import *
import torchvision.transforms as transforms


def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad

def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


def random_resize(images, min_size=288, max_size=448):
    new_size = random.sample(list(range(min_size, max_size + 1, 32)), 1)[0]
    images = F.interpolate(images, size=new_size, mode="nearest")
    return images

class LFFDDataset(Dataset):
    def __init__(self, img_path, transform=None):
        self.img_paths = glob.glob(img_path + "/images/*.jpg")
        self.labels = [label.replace(".jpg", ".xml").replace("images", "labels") for label in self.img_paths]
        self.class_names = ("__background__", "face")
        prior = Priors()
        self.priors = prior()  # center form
        #self.imgW, self.imgH = 640, 640
        self.normalized_labels = False

    def __getitem__(self, idx):
        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(self.img_paths[idx]).convert("RGB"))
        # Handle images with less than three channels
        if len(img.shape) != 3:
            img = img.unsqueeze(0)
            img = img.expand((3, img.shape[1:]))
        _, h, w = img.shape
        h_factor, w_factor = (h, w) if self.normalized_labels else (1, 1)

        # Pad to square resolution
        img, pad = pad_to_square(img, 0)
        _, padded_h, padded_w = img.shape

        label_file = self.labels[idx]
        gt_bboxes, gt_classes = self._get_annotation(idx)
        # img = ToTensor()(img)
        gt_bboxes = torch.tensor(gt_bboxes)
        gt_classes = torch.LongTensor(gt_classes)

        # Extract coordinates
        x1 = w_factor * (gt_bboxes[:, 0] - gt_bboxes[:, 2] / 2)
        y1 = h_factor * (gt_bboxes[:, 1] - gt_bboxes[:, 3] / 2)
        x2 = w_factor * (gt_bboxes[:, 0] + gt_bboxes[:, 2] / 2)
        y2 = h_factor * (gt_bboxes[:, 1] + gt_bboxes[:, 3] / 2)
        # Adjust for added padding
        x1 += pad[0]
        y1 += pad[2]
        x2 += pad[1]
        y2 += pad[3]

        # Returns (x, y, w, h)
        gt_bboxes[:, 0] = ((x1 + x2) / 2) / padded_w
        gt_bboxes[:, 1] = ((y1 + y2) / 2) / padded_h
        gt_bboxes[:, 2] *= w_factor / padded_w
        gt_bboxes[:, 3] *= h_factor / padded_h

        gt_bboxes, gt_classes, ignored = assign_priors(gt_bboxes, gt_classes, self.priors, 0.5)
        locations = convert_boxes_to_locations(gt_bboxes, self.priors, 2)


        return [img, locations, gt_classes, ignored, self.img_paths[idx]]

    def _get_annotation(self, idx):
        annotation_file = self.labels[idx]
        objects = ET.parse(annotation_file).findall("object")
        size = ET.parse(annotation_file).find("size")
        boxes = []
        labels = []
        # is_difficult = []
        for obj in objects:
            class_name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')
            x1 = float(bbox.find('xmin').text) - 1
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1
            imgW = float(size.find('width').text)
            imgH = float(size.find('height').text)
            boxes.append([x1 / imgW, y1 / imgH, x2 / imgW, y2 / imgH])
            labels.append(self.class_names.index(class_name))
        return boxes, labels

    def __len__(self):
        return len(self.img_paths)


if __name__ == '__main__':
    import random
    from data.augmentor import LFFDAug

    transform = LFFDAug(to_tensor=False)
    import cv2 as cv

    datset = LFFDDataset("/home/hp/Data/FaceData/FaceDex")
    # print(datset.labels)
    #
    img, gt_loc, gt_labels, ignored, name= datset[random.choice(range(len(datset)))]
    print(name)
    # image = cv.imread(name)
    # cv.imshow("1", image)
    # cv.waitKey(100)
    # img.show()
    print(ignored.size())
    print(ignored)
    cv_img = np.array(img)
    h,w,_ = cv_img.shape
    cv_img = cv.cvtColor(cv_img, cv.COLOR_RGB2BGR)
    priors = datset.priors
    idx = (gt_labels > 0) & ignored
    loc = convert_locations_to_boxes(gt_loc, datset.priors, 2)
    loc = loc[idx]
    priors = priors[idx]
    label = gt_labels[idx]
    print(loc.size())
    for i in range(priors.size(0)):
        x, y, r = priors[i, :]
        xt, yt, xtl, ytl = loc[i, :]
        # print(x,y,r)
        x = x.item() * w
        y = y.item() * h
        r = r.item() * 640

        xt = xt.item() * w
        yt = yt.item() *h
        xtl = xtl.item() * w
        ytl = ytl.item() * h

        print(x, y, r)
        cv.circle(cv_img, (int(x), int(y)), int(r), (255, 0, 0), 2)

        cv.rectangle(cv_img, (int(xt), int(yt)), (int(xtl), int(ytl)), (0, 255, 0), 2)
    cv.imshow("cv", cv_img)
    cv.waitKey(5000)
