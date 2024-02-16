# Description :  ResNet50 Model to classify crack or no crack
# Date : 12/18/2023 (18)
# Author : Dude
# URLs :  https://www.kaggle.com/code/gxkok21/resnet50-with-pytorch
#
# Problems / Solutions :
#
# Revisions :
#
import torch.nn as nn
import torchvision.models as model_zoo


class CrackDetectionModel(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(CrackDetectionModel, self).__init__()
        self.gradients = None
        self.tensorhook = []
        self.layerhook = []
        self.model = model_zoo.resnet50(pretrained=True)

        # isolate feature blocks
        self.feature_extractor = nn.Sequential(
            self.model.conv1,
            self.model.bn1,
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False
            ),
            self.model.layer1,
            self.model.layer2,
            self.model.layer3,
            self.model.layer4,
        )
        num_fetrs = self.model.fc.in_features

        self.model.fc = nn.Linear(in_features=num_fetrs, out_features=num_classes)

        # for param in self.model.parameters() :
        #    param.requires_grad = False

        # Average Pool Layer
        self.avgpool = self.model.avgpool
        # Classifier
        self.classifier = self.model.fc
        # Gradient Placeholder
        self.gradients = None

    def activations_hook(self, grad):
        self.gradients = grad

    def get_activation_gradient(self):
        return self.gradients

    def get_activations(self, x):
        return self.feature_extractor(x)

    def forward(self, in_x):
        # extract the features
        x = self.feature_extractor(in_x)  # activation maps to use in GradCAM
        if self.training:  # We cannot set hooks if require_grad is false(eval mode)
            h = x.register_hook(self.activations_hook)  # Register activations hook
        x = self.avgpool(x)
        # x=x.view((1,-1))
        x = x.view(x.size(0), -1)
        output = self.classifier(x)
        return output

    # class CrackDetectionModel(nn.Module):
    #     def __init__(self,input_shape, num_classes):
    #         super(CrackDetectionModel,self).__init__()
    #         self.model = model_zoo.resnet50(pretrained=True)
    #         for param in self.model.parameters():
    #             param.requires_grad = False
    #         self.model.fc = nn.Sequential(
    #             nn.Linear(in_features=2048,out_features=num_classes),
    #             nn.Sigmoid())
    #
    #     def forward(self,x):
    #         output = self.model(x)
    #         return  output
    #
    def train_all(self):
        # Un-Freeze the weights of the network
        for param in self.model.parameters():
            param.requires_grad = True
