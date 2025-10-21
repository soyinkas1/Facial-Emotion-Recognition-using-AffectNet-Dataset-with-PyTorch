import pandas as pd
import random
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from torchvision import datasets
from torch.utils.data import DataLoader, TensorDataset, Subset, Dataset
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import sys

# Import custom modules
from utils.common import get_mean_std
from utils.logging import logger
from utils.exception import CustomException

class CustomDataset(Dataset):
    """
    Class to load the dataset and return a PyTorch Dataset object.

    This class supports balanced sampling, allowing users to limit the number of images per label
    for controlled experimentation.
    """
    def __init__(self, data_dir, transform=None, number_of_images=None, balanced_data=False):
        try:
            logger.info('Initializing custom data extraction...')
            
            # Load the full dataset
            self.full_data = datasets.ImageFolder(data_dir, transform=None)
            self.classes = self.full_data.classes
            self.class_to_idx = self.full_data.class_to_idx
            self.idx_to_class = {value: key for key, value in self.class_to_idx.items()}

            self.data_dir = data_dir
            self.transform = transform

            if balanced_data:
                labels = list(self.idx_to_class.keys())

                if number_of_images is None:
                    # Balance dataset based on smallest class size
                    class_counts = pd.Series([sample[1] for sample in self.full_data.samples]).value_counts()
                    samples_per_label = min(class_counts)
                    sampled_indices = []

                    for label in tqdm(labels, desc="Balancing classes"):
                        label_indices = [i for i, sample in enumerate(self.full_data.samples) if sample[1] == label]
                        sampled_indices.extend(random.sample(label_indices, samples_per_label))
                else:
                    # Balanced sampling with specified number of images
                    samples_per_label = max(1, number_of_images // len(labels))
                    sampled_indices = []

                    for label in tqdm(labels, desc="Balancing classes"):
                        label_indices = [i for i, sample in enumerate(self.full_data.samples) if sample[1] == label]
                        sampled_indices.extend(random.sample(label_indices, min(samples_per_label, len(label_indices))))

                    # Adjust to match number_of_images
                    if len(sampled_indices) < number_of_images:
                        additional_indices = random.sample(range(len(self.full_data)), number_of_images - len(sampled_indices))
                        sampled_indices.extend(additional_indices)
                    elif len(sampled_indices) > number_of_images:
                        sampled_indices = random.sample(sampled_indices, number_of_images)

                self.indices = sampled_indices
            else:
                # Random sampling or full dataset
                self.indices = list(range(len(self.full_data))) if number_of_images is None else \
                            random.sample(range(len(self.full_data)), min(number_of_images, len(self.full_data)))

            logger.info('Custom data extraction done.')
        except Exception as e:
            raise CustomException(f"Error initializing CustomDataset: {str(e)}", sys)

    def __len__(self):
        try:
            return len(self.indices)
        except Exception as e:
            raise CustomException(f"Error getting dataset length: {str(e)}", sys)

    def __getitem__(self, idx):
        try:
            sample_idx = self.indices[idx]
            img_path, label = self.full_data.samples[sample_idx]

            # Load image on-demand
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)

            return image, label
        except Exception as e:
            raise CustomException(f"Error getting item at index {idx}: {str(e)}", sys)

class DataETL:
    """
    Class for loading, preprocessing, and transforming datasets efficiently.
    """
    def __init__(self, data_dir, random_state, number_of_images=None, balanced_data=False):
        try:
            self.data_dir = data_dir
            self.random_state = random_state
            self.number_of_images = number_of_images
            self.balanced_data = balanced_data
            self.dataset = None
            self.train_dataset = None
            self.val_dataset = None
            self.test_dataset = None
            self.class_to_idx = None
        except Exception as e:
            raise CustomException(f"Error initializing DataETL: {str(e)}", sys)

    def extract_data(self, transform):
        try:
            logger.info('Extracting data......')
            self.dataset = CustomDataset(self.data_dir, transform=transform,
                                      number_of_images=self.number_of_images,
                                      balanced_data=self.balanced_data)
            self.class_to_idx = self.dataset.class_to_idx
            logger.info('Data extraction done......')
        except Exception as e:
            raise CustomException(f"Error extracting data: {str(e)}", sys)

    def split_data(self):
        try:
            logger.info('Splitting data started.......')
            indices = np.arange(len(self.dataset))
            labels = [self.dataset.full_data.samples[self.dataset.indices[i]][1] for i in indices]

            train_idx, temp_idx, y_train, y_temp = train_test_split(indices, labels, test_size=0.2,
                                                                  stratify=labels, random_state=self.random_state)
            val_idx, test_idx, y_val, y_test = train_test_split(temp_idx, y_temp, test_size=0.5,
                                                              stratify=y_temp, random_state=self.random_state)

            self.train_dataset = Subset(self.dataset, train_idx)
            self.val_dataset = Subset(self.dataset, val_idx)
            self.test_dataset = Subset(self.dataset, test_idx)
            logger.info('Splitting data completed.......')
        except Exception as e:
            raise CustomException(f"Error splitting data: {str(e)}", sys)

    def transform_load(self, dataset_specific_norm=True, batch_size=32, aug_class=None):
        try:
            logger.info('Transformation started......')

            # Define normalization transformation
            if dataset_specific_norm:
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                ])
                self.dataset = CustomDataset(self.data_dir, transform=transform,
                                      number_of_images=self.number_of_images,
                                      balanced_data=self.balanced_data)
                loader = DataLoader(self.dataset, batch_size=32, shuffle=True, num_workers=2)
                mean, std = get_mean_std(loader)
            else:
                mean = [0.5, 0.5, 0.5]
                std = [0.5, 0.5, 0.5]

            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])

            # Load dataset once with final transform
            self.extract_data(transform)
            self.split_data()

            if aug_class:
                logger.info('Applying data augmentation to training data......')

                # Define Preprocessing and Augmentation Transforms
                data_aug_transformation = transforms.Compose([
                    transforms.RandomRotation(20),
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std)
                ])
                
                train_indices = self.train_dataset.indices
                train_labels = [self.dataset.full_data.samples[self.dataset.indices[i]][1] for i in train_indices]
                class_counts = pd.Series(train_labels).value_counts()
                target_samples = max(class_counts)
                
                aug_class_idx = [self.class_to_idx[cls] for cls in aug_class if cls in self.class_to_idx]
                if not aug_class_idx:
                    logger.info("Warning: No valid classes in aug_class for augmentation")
                    return self.train_dataset, self.val_dataset, self.test_dataset

                augmented_data = []
                augmented_labels = []

                for cls in tqdm(aug_class_idx, desc="Augmenting classes"):
                    cls_indices = [i for i, lbl in enumerate(train_labels) if lbl == cls]
                    current_samples = len(cls_indices)
                    aug_samples = target_samples - current_samples
                    if aug_samples <= 0:
                        continue

                    for _ in tqdm(range(aug_samples), desc=f"Augmenting class {cls}"):
                        idx = train_indices[cls_indices[np.random.randint(0, current_samples)]]
                        img_path, label = self.dataset.full_data.samples[self.dataset.indices[idx]]
                        img = Image.open(img_path).convert("RGB")
                        img = data_aug_transformation(img)
                        augmented_data.append(img)
                        augmented_labels.append(label)

                if augmented_data:
                    orig_data = [self.dataset[i][0] for i in train_indices]
                    orig_labels = [self.dataset[i][1] for i in train_indices]
                    augmented_data = torch.stack(augmented_data)
                    augmented_labels = torch.tensor(augmented_labels, dtype=torch.long)
                    all_data = torch.cat([torch.stack(orig_data), augmented_data])
                    all_labels = torch.cat([torch.tensor(orig_labels, dtype=torch.long), augmented_labels])
                    self.train_dataset = TensorDataset(all_data, all_labels)

            logger.info(f'Train data count: {len(self.train_dataset)}')
            logger.info(f'Validation data count: {len(self.val_dataset)}')
            logger.info(f'Test data count: {len(self.test_dataset)}')
            logger.info('Transformation completed')

            return self.train_dataset, self.val_dataset, self.test_dataset

        except Exception as e:
            raise CustomException(f"Error during transformation: {str(e)}", sys)

# if __name__ == "__main__":
#     try:
#         data_etl = DataETL('data_new', 42, 200, False)
#         train, val, test = data_etl.transform_load() 
#         print(type(train))
#     except Exception as e:
#         raise CustomException(f"Error in main execution: {str(e)}", sys)