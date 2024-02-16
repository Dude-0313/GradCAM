# Description : Load SDNet2018 Dataset
# Date : 11/14/2023 (14)
# Author : Dude
# URLs :
# SDNET2018: An annotated image dataset for non-contact concrete crack detection using deep convolutional neural networks
#  https://www.kaggle.com/datasets/harishmulchandani2/sdnet2018?resource=download
# Problems / Solutions :
#
# Revisions :
#
import os
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image


class SD2018(Dataset):
    def __init__(self, data_path, pos_dir, neg_dir, transform=None):
        self.datapath = data_path
        self.pos_dir = pos_dir
        self.neg_dir = neg_dir
        self.transform = transform
        self.labels = []
        # self.labels_df, self.labels_map = load_labels(
        #     os.path.join(data_path, labels_csv)
        # )
        self.pos_filelist = os.listdir(os.path.join(data_path, pos_dir))
        self.neg_filelist = os.listdir(os.path.join(data_path, neg_dir))
        self.full_filelist = self.pos_filelist + self.neg_filelist
        image = read_image(os.path.join(data_path, pos_dir, self.pos_filelist[0]))
        print(image.shape)
        self.image_shape = image.shape

    def __len__(self):
        return len(self.full_filelist)

    def __getitem__(self, idx):
        if idx < len(self.pos_filelist):
            image = read_image(
                os.path.join(self.datapath, self.pos_dir, self.full_filelist[idx])
            )
            class_idx = 1
        else:
            image = read_image(
                os.path.join(self.datapath, self.neg_dir, self.full_filelist[idx])
            )
            class_idx = 0

        if self.transform:
            return self.transform(image), torch.as_tensor(class_idx, dtype=torch.int64)
        else:
            return image, torch.tensor([class_idx])

    def get_classes(self):
        return {"Positive": 1, "Negative": 0}

    def get_shape(self):
        return self.image_shape
