import os
from typing_extensions import Self
import numpy as np

import albumentations as A
from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision.io import read_image
from train_test_split_script import Data_Split
from albumentations.pytorch.transforms import ToTensorV2


class Augmentation_Data_Basics(Data_Split):
    def __init__(self, augment_probability):
        super().__init__(0.5)
        self.augment_probability = augment_probability
        self.l1 = []
        self.count = 0
        
    def augment_data(self):
        split_object = Data_Split(self.split_ratio)
        train_data, validation_data = split_object.split_data()
        transforms_on_img = A.Compose(
            [
                A.HorizontalFlip(self.augment_probability),
                ToTensorV2(),
            ]
        )
        # print(len(train_data))
        for img in train_data:
            image = np.array(img.permute(1, 2, 0))
            transformed_image = transforms_on_img(image=image)['image']
            self.l1.append(transformed_image)
            # transform = T.ToPILImage()
            # v_image = transform(transformed_image)
            # self.count += 1
            # v_image = v_image.save(f"fastapi-augmented-folder/{self.count}.jpg")
        # print(len(self.l1))
        return self.l1


obj1 = Augmentation_Data_Basics(0.5)
obj1.augment_data()