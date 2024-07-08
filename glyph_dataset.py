from torch.utils.data import DataLoader, Dataset
import os
import matplotlib.image as mpimg
import numpy as np
import json

class Glyph_Dataset(Dataset):
    def __init__(self, dataset_path, label_encoder_file, augmentation=None, preprocessing=None, transform=None, oversample_factor=0):
        self.dataset_path = dataset_path
        self.filenames = os.listdir(dataset_path)
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.transform = transform
        self.oversample_factor = oversample_factor

        with open(label_encoder_file, 'r') as f:
            self.label_encoder = json.load(f)
            
        # Balance the dataset by oversampling minority classes
        self.balanced_filenames = self.balance_dataset()

    def __len__(self):
        return len(self.balanced_filenames)
    
    def balance_dataset(self):
        # If oversample_factor is 0, do not perform any oversampling
        if self.oversample_factor == 0:
            return self.filenames

        # Count the occurrences of each class
        class_counts = {}
        for filename in self.filenames:
            last_underscore_index = filename.rfind('_')
            label_str = filename[last_underscore_index + 1:filename.rfind('.')]
            label_int = self.label_encoder[label_str]
            class_counts[label_int] = class_counts.get(label_int, 0) + 1

        max_count = max(class_counts.values())
        target_count = max_count * self.oversample_factor
        balanced_filenames = []

        # Oversample each class to match the target count
        for filename in self.filenames:
            last_underscore_index = filename.rfind('_')
            label_str = filename[last_underscore_index + 1:filename.rfind('.')]
            label_int = self.label_encoder[label_str]
            count = class_counts[label_int]

            # Only oversample classes below the target count
            if count < target_count:
                multiplier = int(target_count // count)
                additional_samples = int((target_count / count - multiplier) * self.oversample_factor)
                balanced_filenames.extend([filename] * (multiplier + additional_samples))
            else:
                balanced_filenames.append(filename)

        return balanced_filenames

    def __getitem__(self, index):
        actual_index = index % len(self.balanced_filenames)
        image_path = os.path.join(self.dataset_path, self.balanced_filenames[actual_index])
        image = mpimg.imread(image_path)

        last_underscore_index = self.balanced_filenames[actual_index].rfind('_')
        label_str = self.balanced_filenames[actual_index][last_underscore_index + 1:self.balanced_filenames[actual_index].rfind('.')]
        label_int = self.label_encoder[label_str]

        if self.preprocessing:
            preprocessed = self.preprocessing(image)
            image = preprocessed["image"]

        if self.augmentation:
            augmented = self.augmentation(image=image)['image']
            
        if self.transform:
            image = self.transform(image)

        return image, label_int