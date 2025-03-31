import os
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class NIH_Dataset(Dataset):
    def __init__(self, data_dir, transform=None, data_type='train'):
        """
        Args:
            data_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            data_type (string): Type of data to load ('train' or 'test').
        """
        self.data_dir = data_dir
        self.transform = transform
        self.data_type = data_type
        self.train_list_file = os.path.join(data_dir, 'train_val_list.txt')
        self.test_list_file = os.path.join(data_dir, 'test_list.txt')
        self.ground_truth_file = os.path.join(data_dir, 'Data_Entry_2017.csv')
        self.classes = [
            "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
            "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
            "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia", "No Finding"
        ]

        # image folders
        self.image_dir = [
            self.data_dir+'/'+dir+'/images' for dir in os.listdir(data_dir)
            if dir.startswith('images_') and 
            os.path.isdir(os.path.join(data_dir, dir, 'images'))
        ]

        # image files
        self.image_files = [
            os.path.join(dir, img) 
            for dir in self.image_dir 
            for img in os.listdir(dir)
            if img.endswith('.png')
        ]

        # Filter image files based on data_type
        with open(self.train_list_file, 'r') as f:
            train_files = [line.strip() for line in f.readlines()]
        with open(self.test_list_file, 'r') as f:
            test_files = [line.strip() for line in f.readlines()]
        
        if self.data_type == 'train':
            self.image_files = [f for f in self.image_files if os.path.basename(f) in train_files]
            print(f"Number of training images: {len(self.image_files)}")
        elif self.data_type == 'test':
            self.image_files = [f for f in self.image_files if os.path.basename(f) in test_files]
            print(f"Number of testing images: {len(self.image_files)}")
        else:
            raise ValueError("data_type must be 'train' or 'test'")

        # read ground truth labels
        ground_truth_df = pd.read_csv(self.ground_truth_file)
        self.ground_truth = {
            row['Image Index']: row['Finding Labels'].split('|') 
            for _, row in ground_truth_df.iterrows()
        }

    def encode_labels(self, labels):
        """
        Encode the labels into a binary vector.
        """
        label_vector = np.zeros(len(self.classes), dtype=int)
        for label in labels:
            if label in self.classes:
                index = self.classes.index(label)
                label_vector[index] = 1
        return label_vector

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path)
    
        if self.transform:
            image = self.transform(image)

        labels = self.ground_truth.get(os.path.basename(img_path), [])
        encoded_labels = self.encode_labels(labels)

        return image, labels, encoded_labels