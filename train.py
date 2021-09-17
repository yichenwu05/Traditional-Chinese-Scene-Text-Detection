import os
import cv2
import json
import torch
import pickle
import numpy as np
import torch.optim as optim
import torch.utils.data
import torchvision
from torchvision.models.detection import FasterRCNN, MaskRCNN
from torchvision.models.detection.backbone_utils import (BackboneWithFPN,
                                                         resnet_fpn_backbone)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from PIL import Image
from utils import utils
import utils.transforms as T
from utils.engine import train_one_epoch, evaluate
import PIL
PIL.Image.MAX_IMAGE_PIXELS = 1e10

gpu = 0
device = torch.device('cuda:%d' % (
    gpu) if torch.cuda.is_available() else 'cpu')


class SceneTextDataset(torch.utils.data.Dataset):
    def __init__(self, folder, transforms=None):

        
        self.folder = folder
        self.transforms = transforms
        
        with open('./dataset/{}/index_map.pickle'.format(self.folder), 'rb') as file:
            self.index_map = pickle.load(file)
        file.close()


        self.total = len(self.index_map)

    def __getitem__(self, idx):
        

        file_name = self.index_map[idx]
        
        img = Image.open('./dataset/{}/img/'.format(self.folder)+file_name+'.jpg').convert("RGB")
        
        with open('./dataset/{}/gt/'.format(self.folder)+file_name+'.pickle', 'rb') as file:
            gt = pickle.load(file)
        file.close()
        
        masks = Image.open('./dataset/{}/mask/'.format(self.folder)+file_name+'.png')

        masks = np.array(masks)
        masks = np.reshape(masks, (gt['num_objs'], int(masks.shape[0]/gt['num_objs']), masks.shape[1]))
        masks = masks.astype(np.uint8)
        boxes = gt['boxes']
        areas = gt['areas']
        num_objs = gt['num_objs']
        
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        
        image_id = torch.tensor([idx])
        area = torch.tensor(areas)*1.0
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return self.total


def maskrcnn_resnet152_fpn(progress=True, num_classes=2, pretrained_backbone=True, **kwargs):
    anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
    aspect_ratios = ((0.3, 0.5, 1.0, 1.5, 2.0, 2.5, 3),) * len(anchor_sizes)
    rpn_anchor_generator = AnchorGenerator(
        anchor_sizes, aspect_ratios
    )
    backbone = resnet_fpn_backbone("resnet152", pretrained_backbone)
    model = MaskRCNN(backbone, num_classes, rpn_anchor_generator=rpn_anchor_generator, **kwargs)
    return model


def get_instance_segmentation_model(num_classes, load_model=None):
    # load an instance segmentation model pre-trained on COCO
    # model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    # model = maskrcnn_resnet50_fpn(num_classes=num_classes)
    model = maskrcnn_resnet152_fpn(num_classes=num_classes)
    # model = maskrcnn_resnext101_328_fpn(num_classes=num_classes)
    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256

    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    return model


def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))

    return T.Compose(transforms)



if __name__ == '__main__':

    dataset_train = SceneTextDataset('train', get_transform(train=True))
    dataset_dev = SceneTextDataset('dev', get_transform(train=False))

    # define training and validation data loaders
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=6, shuffle=True, num_workers=0,
        collate_fn=utils.collate_fn)

    data_loader_dev = torch.utils.data.DataLoader(
        dataset_dev, batch_size=16, shuffle=False, num_workers=0,
        collate_fn=utils.collate_fn)

    # the dataset has two classes only - background and person
    num_classes = 2

    # get the model using the helper function
    model = get_instance_segmentation_model(num_classes)
    # move model to the right device
    model.to(device)

    for param in model.parameters():
        param.requires_grad = True

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)


    # the learning rate scheduler decreases the learning rate by 10x every 5 epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=5,
                                                   gamma=0.1)


    num_epochs = 3
    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq=500)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_dev, device=device)
        # save model weight 
        torch.save(model.state_dict(), './model/resnet152_yours_{}.ckpt'.format(epoch+1))