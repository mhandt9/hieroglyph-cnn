from torch.utils.data import DataLoader, Dataset
import os
import matplotlib.image as mpimg
import numpy as np
import json

class Glyph_Dataset(Dataset):
    """ Glyph Dataset, Read images, apply augmentation and preprocessing transformations
    Args:
        dataset_path(str): the root path of the dataset
        augmentation: perform augmentation: rotating, blurring, etc
        preprocessing: perform necessary preprocessing steps
    """
class Glyph_Dataset(Dataset):
    def __init__(self, dataset_path, label_encoder_file, augmentation=None, preprocessing=None):
        self.dataset_path = dataset_path
        self.filenames = os.listdir(dataset_path)
        self.augmentation = augmentation
        self.preprocessing = preprocessing

        with open(label_encoder_file, 'r') as f:
            self.label_encoder = json.load(f)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        image_path = os.path.join(self.dataset_path, self.filenames[index])
        image = mpimg.imread(image_path)

        last_underscore_index = self.filenames[index].rfind('_')
        label_str = self.filenames[index][last_underscore_index+1:self.filenames[index].rfind('.')]
        label_int = self.label_encoder[label_str]

        if self.preprocessing:
            image = self.preprocessing(image)

        if self.augmentation:
            image = self.augmentation(image)

        return image, label_int