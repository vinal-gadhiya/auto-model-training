from genericpath import isfile
from itertools import count
import os
import matplotlib.pyplot as plt

class number_of_images:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_count = 0
    
    def get_number_of_images(self):
        for file_ in os.listdir(self.root_dir):
            if os.path.isfile(os.path.join(self.root_dir, file_)):
                self.image_count += 1
        return self.image_count