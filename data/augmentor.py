'''
@Descripttion: This is menger's demo,which is only for reference
@version:
@Author: menger
@Date: 2020-12-8
@LastEditors:
@LastEditTime:
'''
import cv2 as cv
import numpy as np
from PIL import Image
from torchvision import transforms

from albumentations import (

    HorizontalFlip,
    Resize,
    RandomSizedBBoxSafeCrop,
    Compose,
    RandomSunFlare,
    RandomShadow,
    RandomBrightness,
    RandomContrast,
    RandomCrop,
    GaussianBlur,
    CenterCrop,
    SmallestMaxSize,
    PadIfNeeded,
    LongestMaxSize
)
from torchvision.transforms import ToTensor


class LFFDAug(object):
    def __init__(self, to_tensor=True):
        augs = [
            LongestMaxSize(640),
            PadIfNeeded(640, 640, cv.BORDER_CONSTANT, value=0),
            # # RandomSizedBBoxSafeCrop(height = 300,width = 300),
            RandomBrightness(p=0.5),
            RandomContrast(p=0.5),

            # RandomSunFlare(p=0.5, flare_roi=(0, 0, 1, 0.5), angle_lower=0.5,src_radius= 150),
            RandomShadow(p=0.5, num_shadows_lower=1, num_shadows_upper=1,
                         shadow_dimension=5, shadow_roi=(0, 0.5, 1, 1)),
            HorizontalFlip(p=0.5),
            GaussianBlur(p=0.5),
        ]
        self.transform = Compose(augs,
                                 bbox_params={"format": "albumentations", "min_area": 0, "min_visibility": 0.2,
                                              'label_fields': ['category_id']}
                                 )
        self.to_Tensor = to_tensor

    def __call__(self, cv_img, boxes=None, labels=None):
        auged = self.transform(image=cv_img, bboxes=boxes, category_id=labels)
        if self.to_Tensor:
            return ToTensor()(auged["image"]), auged["bboxes"], auged["category_id"]
        else:
            return (auged["image"]), auged["bboxes"], auged["category_id"]


if __name__ == "__main__":
    aug = LFFDAug()
    img = cv.imread("/home/hp/Data/FaceData/FaceDex/images/0--Parade_0_Parade_marchingband_1_5.jpg")
    cv.imshow("img", img)
    cv.waitKey(1000)
    boxes = [[0.2, 0.2, 0.5, 0.5], [0, 0, 0.1, 0.1]]
    labels = [0, 1]
    img, boxes, labels = aug(img, boxes, labels)
    img = transforms.ToPILImage()(img.float())
    print(img)
    img.show()
    img = cv.cvtColor(np.asarray(img), cv.COLOR_RGB2BGR)
    print(img.shape)


    for box in boxes:
        xmin, ymin, xmax, ymax = box
        xmin *= 640
        ymin *= 640
        xmax *= 640
        ymax *= 640
        cv.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 0, 0), 2)
    cv.imshow("img", img)
    cv.waitKey(0)
