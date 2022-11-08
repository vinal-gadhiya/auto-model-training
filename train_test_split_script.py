import os

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision.io import read_image
import torchvision.transforms as T
from PIL import Image


class LoadDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.list_files = os.listdir(self.root_dir)
        # print(self.list_files)

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, index):
        img_file = self.list_files[index]
        img_path = os.path.join(self.root_dir, img_file)
        image = read_image(img_path)
        return image
        

class Data_Split(LoadDataset):
    def __init__(self, split_ratio):
        super().__init__("fastapi-test-folder")
        self.split_ratio = split_ratio
        self.root_dir = "fastapi-test-folder"
        self.count = 0

    def split_data(self):
        self.img_data = LoadDataset(self.root_dir)
        train_data, validation_data = train_test_split(self.img_data, train_size=self.split_ratio, random_state=42)

        '''Required only if we want to create a new folder and want to save train and test images individually'''
        # if not os.path.isdir("fastapi-test-folder-splitted/train"):
        #     os.mkdir("fastapi-test-folder-splitted/train")
        #     os.mkdir("fastapi-test-folder-splitted/validate")
        # for t_img in train_data:
        #     transform = T.ToPILImage()
        #     t_image = transform(t_img)
        #     self.count += 1
        #     t_image = t_image.save(f"fastapi-test-folder-splitted/train/{self.count}.jpg")
        # for v_img in validation_data:
        #     transform = T.ToPILImage()
        #     v_image = transform(v_img)
        #     self.count += 1
        #     v_image = v_image.save(f"fastapi-test-folder-splitted/validate/{self.count}.jpg")

        return train_data, validation_data


# obj1 = Data_Split(0.5)
# obj1.split_data()