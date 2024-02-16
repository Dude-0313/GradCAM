# Description : Using GRAD Cam for highlighting area of detection
# Date : 11/28/2023 (28)
# Author : Dude
# URLs :
#  https://medium.com/the-owl/gradcam-in-pytorch-7b700caa79e5
# Problems / Solutions :
#  P1: Cuda Out of Error while evaluating for confusion matrix
#   (RuntimeError: CUDA out of memory. Tried to allocate 20.00 MiB (GPU 0; 16.00 GiB total capacity; 15.24 GiB already allocated)
#  S1 :Caused by  attempting  to store all tensors in GPU memory
# - Moved cumulative tensors to CPU
# - Reduced test batch size, test dataset size, used cuda garbage collection and clear cache
# Revisions :
#
import random
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from CrackResNet50 import CrackDetectionModel

# from GradCAM_Resnet50 import GradCamResnet50
from PlotTrainingCurves import plot_training_curves, plot_cm
from SD2018Dataset import SD2018
from TrainTestEval import train_classification_model

DATA_PATH = "C:\\Kuljeet\\Datasets\\SDNET2018\\W"
POS_DIR = "CW"
NEG_DIR = "UW"
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.15


def get_gradcam(model, image, label, size):
    label.backward()
    gradients = model.get_activation_gradient()
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])  # a1, a2, a3 .. ak
    activations = model.get_activations(image).detach()  # A1, A2, A3 ... Ak

    print(pooled_gradients.shape)
    print(activations.shape)
    for i in range(activations.shape[1]):
        activations[:, i, :, :] *= pooled_gradients[i]

    heatmap = torch.mean(activations, dim=1).squeeze().cpu()  # remove batch dimenstion
    heatmap = nn.ReLU()(heatmap)  # run ReLU on output to remove unwanted activations
    heatmap /= torch.max(heatmap)
    heatmap = cv2.resize(heatmap.numpy(), (size, size))  # convert back to image shape

    return heatmap


def plot_heatmap(denorm_image, data_classes, pred, heatmap):
    fig, (ax1, ax2, ax3) = plt.subplots(figsize=(20, 20), ncols=3)

    classes = np.array(list(data_classes.keys()))
    ps = torch.nn.Softmax(dim=1)(pred).cpu().detach().numpy()
    ax1.imshow(denorm_image)

    ax2.barh(classes, ps[0])
    ax2.set_aspect(0.1)
    ax2.set_yticks(classes)
    ax2.set_yticklabels(classes)
    ax2.set_title("Predicted Class")
    ax2.set_xlim(0, 1.1)

    ax3.imshow(denorm_image)
    ax3.imshow(heatmap, cmap="magma", alpha=0.7)
    plt.show(block=True)

def LoadDatasets():
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    sd2018_dataset = SD2018(DATA_PATH, POS_DIR, NEG_DIR, transform=transform)
    input_shape = sd2018_dataset.get_shape()
    data_classes = sd2018_dataset.get_classes()
    print(data_classes.keys())
    # input_shape = (3, 128, 235)
    # data_classes = 8
    train_size = int(TRAIN_SPLIT * len(sd2018_dataset))
    val_size = int(VAL_SPLIT * len(sd2018_dataset))
    test_size = len(sd2018_dataset) - val_size - train_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        sd2018_dataset, [train_size, val_size, test_size]
    )
    data_loaders = {
        "train": DataLoader(dataset=train_dataset, batch_size=8, shuffle=True),
        "val": DataLoader(dataset=val_dataset, batch_size=8, shuffle=True),
        "test": DataLoader(dataset=test_dataset, batch_size=4, shuffle=True),
    }
    dataset_sizes = {
        "train": len(train_dataset),
        "val": len(val_dataset),
        "test": len(test_dataset),
    }
    print("Dataset sizes :")
    print(dataset_sizes)
    return data_loaders, dataset_sizes, input_shape, data_classes


# random split
SEED = random.randint(1, 100)
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
# parameters
learning_rate = 0.01
num_epochs = 10
# use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load custom respiratory dataset
dataloaders, dataset_sizes, input_shape, data_classes = LoadDatasets()

# load model
# model = GradCamResnet50(input_shape,len(data_classes))
model = CrackDetectionModel(input_shape, len(data_classes)).to(device)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.90)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)


# # Make sure you save the training curves along the way for visualization afterwards!
model, training_curves = train_classification_model(
    device,
    model,
    dataloaders,
    dataset_sizes,
    criterion,
    optimizer,
    scheduler,
    num_epochs=num_epochs,
)

# Unfreeze all layers and tune
model.train_all()
# # Make sure you save the training curves along the way for visualization afterwards!
model, training_curves = train_classification_model(
    device,
    model,
    dataloaders,
    dataset_sizes,
    criterion,
    optimizer,
    scheduler,
    num_epochs=num_epochs,
    save_best_model=True,
)

# plot training curves
plot_training_curves(training_curves, phases=["train", "val", "test"])

# plot confusion matrix
rep = plot_cm(model, device, dataloaders, list(data_classes.keys()), phase="test")

# print classification report
print("============== Classification Report ==============")
print(rep)
print("==================================================")

# Show GradCAM results
for image, label in dataloaders["val"].dataset:
    if label > 0:
        break
# denormalize using ImageNet values
denorm_image = image.permute(1, 2, 0) * np.array((0.229, 0.224, 0.225)) + np.array(
    (0.485, 0.456, 0.406)
)  # denormalize using ImageNet values
image = image.unsqueeze(0).to(device)

pred = model(image)
print(pred)
heatmap = get_gradcam(
    model, image, pred[0][1], size=256
)  # select pred list and then the label in it
plot_heatmap(denorm_image, data_classes, pred, heatmap)
