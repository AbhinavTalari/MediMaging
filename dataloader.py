import os
import cv2
import torch
import torchvision
import numpy as np
import pandas as pd
import torch.utils.data as data
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import train_test_split
import logging
from collections import OrderedDict

logger = logging.getLogger(__name__)


class ChestXRayDataLoader(data.Dataset):
    def __init__(self, path, resize_shape=256, phase="train", transforms=None):
        self.path = path
        self.phase = phase
        self.label_map = OrderedDict({"normal": 0, "pneumonia": 1, "COVID-19": 2})

        data = pd.read_csv(os.path.join(path, phase + ".csv"))

        self.filenames = data["image_names"].to_numpy()
        self.labels = data["label"].to_numpy()

        self.names = []
        self.disease_labels = []
        self.resize_shape = (resize_shape, resize_shape)
        self.transforms = transforms

        try:
            if self.phase == "train":
                for disease in list(self.label_map.keys()):
                    image_names = self.filenames[self.labels == disease][:300]
                    label_names = self.labels[self.labels == disease][:300]
                    self.names.append(image_names)
                    self.disease_labels.append(label_names)

                self.names = np.concatenate(self.names, axis=0)
                self.disease_labels = np.concatenate(self.disease_labels, axis=0)
            else:
                self.names = data["image_names"].to_numpy()
                self.disease_labels = data["label"].to_numpy()
        except Exception as exc:
            logger.error(f"{exc}")

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        img = cv2.imread(os.path.join(self.path, self.phase, self.names[index]))
        img = cv2.resize(img, self.resize_shape)
        img = np.array(img, dtype=np.float32)
        img = img / 255.0
        img = img[:, :, (2, 1, 0)]
        img = torchvision.transforms.ToTensor()(img)

        label = self.label_map[self.disease_labels[index]]
        if self.transforms:
            img = self.transforms(img)

        return img.float(), torch.FloatTensor([label])


def get_dataloader(params, resize_shape, train_transforms=None):
    logger.info(f"Train transforms: {train_transforms}")
    try:
        if params.dataset_name == "disease_data":
            dataset = ChestXRayDataLoader(
                "./data/", resize_shape, "train", train_transforms
            )
            train_idx, valid_idx = train_test_split(
                np.arange(len(dataset.disease_labels)),
                test_size=params.validation_split,
                shuffle=True,
                stratify=dataset.disease_labels,
            )

            train_sampler = SubsetRandomSampler(train_idx)
            valid_sampler = SubsetRandomSampler(valid_idx)

            train_loader = DataLoader(
                dataset,
                batch_size=params.train_batch_size,
                sampler=train_sampler,
                num_workers=params.num_workers,
            )
            valid_loader = DataLoader(
                dataset,
                batch_size=params.train_batch_size,
                sampler=valid_sampler,
                num_workers=params.num_workers,
            )

            test_dataset = ChestXRayDataLoader("./data/", phase="test")
            test_loader = DataLoader(
                test_dataset,
                batch_size=params.test_batch_size,
                num_workers=params.num_workers,
            )

            return {
                "train_loader": train_loader,
                "valid_loader": valid_loader,
                "test_loader": test_loader,
                "label_map": dataset.label_map,
            }
        else:
            logger.error(f"The dataset, {params.dataset_name} is not yet handled!")

    except Exception as exc:
        logger.error(f"{exc}")
