import os
import cv2
import torch
import numpy as np
import torch.utils.data
import torchvision
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import (BackboneWithFPN,
                                                         resnet_fpn_backbone)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from PIL import Image, ExifTags
from utils import utils
import utils.transforms as T
from utils.engine import evaluate
from utils.nms import soft_nms
from tqdm import tqdm
import pandas as pd
import time


gpu = 0
device = torch.device('cuda:%d' % (
    gpu) if torch.cuda.is_available() else 'cpu')
    
    
class SignTestDataset(torch.utils.data.Dataset):
    def __init__(self, folder, transforms=None):
        self.folder = folder
        self.transforms = transforms
        self.imgs = os.listdir('./{}/img'.format(self.folder))
    
    def __getitem__(self, idx):
        
        default = 3001 if 'Private' in self.folder else 1
        
        img = Image.open('./{}/img/img_{}.jpg'.format(self.folder, idx+default))
        img = self.IsRotate(img)
            
        target = {}
        target["image_id"] = idx+default
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target
    
    def IsRotate(self, img): # 返回false表示存在旋转情况
        try:
            for orientation in ExifTags.TAGS.keys() :
                if ExifTags.TAGS[orientation]=='Orientation' :
                    img2 = img.rotate(0, expand = True)
                    break
            exif=dict(img._getexif().items())
            if  exif[orientation] == 3 :
                img2=img.rotate(180, expand = True)
            elif exif[orientation] == 6 :
                img2=img.rotate(270, expand = True)
            elif exif[orientation] == 8 :
                img2=img.rotate(90, expand = True)

            return img2.convert("RGB")
        except:
            return img.convert("RGB")

    def __len__(self):
        return len(self.imgs)
    
    
def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))

    return T.Compose(transforms)
    
    
    
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

    model = maskrcnn_resnet152_fpn(num_classes=num_classes)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256

    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    if load_model:
        model.load_state_dict(torch.load('./model/{}.ckpt'.format(load_model), map_location=device), strict=False)
    return model
    
    
num_classes = 2
# get the model using the helper function
model = get_instance_segmentation_model(num_classes, 'resnet152_12')
# move model to the right device
model.to(device)



def mask_bb(_mask):
    v = np.sum(_mask, axis=1)
    h = np.sum(_mask, axis=0)
    
    _xmin = (h!=0).argmax(axis=0)
    _xmax = len(h) - (h[::-1]!=0).argmax(axis=0) - 1
    
    _ymin = (v!=0).argmax(axis=0)
    _ymax = len(v) - (v[::-1]!=0).argmax(axis=0) - 1
    
    
    return [_xmin, _ymin, _xmax, _ymax]

def to_dets_score(boxes, scores, threshold=[0.5, 0.7, 0.9]):
    dets, areas = [], []
    for i in range(len(boxes)):
        dets.append(boxes[i] + [scores[i]])
        areas.append((boxes[i][2]-boxes[i][0])*(boxes[i][3]-boxes[i][1]))
    dets = np.array(dets)
    result = soft_nms(dets, 0.5)
    valid_index = []
    valid_areas = []
    for i in result:
        if areas[i] < 1024:
            if scores[i] >= threshold[0]:
                valid_index.append(i)
                valid_areas.append(areas[i])
        elif areas[i] < 9216:
            if scores[i] >= threshold[1]:
                valid_index.append(i)
                valid_areas.append(areas[i])
        else:
            if scores[i] >= threshold[2]:
                valid_index.append(i)
                valid_areas.append(areas[i])    
    return valid_index, valid_areas
    
    
def predict_points(boxes, masks, scores, threshold=[0.5, 0.7, 0.9]):

    ### Select predict boxes ( Soft NMS ) ###
    valid_boxes, valid_areas = to_dets_score(boxes.tolist(), scores.tolist(), threshold)
    
    res = []
    s = 0.3
    for kk, index in enumerate(valid_boxes):
        
        mask = masks[index]
        mask = mask[0, :, :]
        mask[mask>=s]=1
        mask[mask<s]=0
        xmin, ymin, xmax, ymax = mask_bb(mask)
        bb_mask = mask[ymin: ymax+1, xmin:xmax+1].copy()
        seg_area = np.sum(bb_mask)
        bb_area = (xmax-xmin)*(ymax-ymin)
        
        mask = np.expand_dims(mask, 2)
        mask = mask.repeat(3, 2)
        mask = mask.astype(np.uint8)
        
        ### Find minimum rectangle area of mask ###
        gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        cnt, _ = cv2.findContours(gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rect = cv2.minAreaRect(cnt[np.argmax([len(x) for x in cnt])])
        box_points = cv2.boxPoints(rect).astype(int)
        x0, y0, x1, y1, x2, y2, x3, y3, score = box_points[0][0], box_points[0][1], box_points[1][0], box_points[1][1], box_points[2][0], box_points[2][1], box_points[3][0], box_points[3][1], scores[index]

        if seg_area/bb_area < 1:
            bb_points = [(box_points[0][0], box_points[0][1]), (box_points[1][0], box_points[1][1]), (box_points[2][0], box_points[2][1]), (box_points[3][0], box_points[3][1])]     
            is_one = np.where(bb_mask == 1)
            seg_points = []
            for i in range(len(is_one[0])):
                seg_points.append((is_one[1][i]+xmin, is_one[0][i]+ymin))
            poly_points = []
            seg_points = np.array(seg_points)
            for p1 in bb_points:
                p1 = np.array([p1]*len(seg_points))
                dists = np.sum((seg_points - p1)**2, 1)
                new_points = seg_points[np.argmin(dists)].tolist()
                p1 = p1[0].tolist()
                new_points = [round(new_points[0]), round(new_points[1])]
                poly_points.append(new_points)
                
            x0, y0 = poly_points[0][0], poly_points[0][1]
            x1, y1 = poly_points[1][0], poly_points[1][1]
            x2, y2 = poly_points[2][0], poly_points[2][1]
            x3, y3 = poly_points[3][0], poly_points[3][1]
            
        res.append([x0, y0, x1, y1, x2, y2, x3, y3, score])

    return res
    
    
def batch_inference(data_loader):
    prediction_boxes = []
    prediction_masks = []
    prediction_scores = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(data_loader):
            imgs = list(img.to(device) for img in batch[0])
            predictions = model(imgs)
            for pred in predictions:
                box = pred['boxes'].detach().cpu().numpy()
                mask = pred['masks'].detach().cpu().numpy()
                score = pred['scores'].detach().cpu().numpy()
                prediction_boxes.append(box)
                prediction_masks.append(mask)
                prediction_scores.append(score)
    return prediction_boxes, prediction_masks, prediction_scores   
    

if __name__ == '__main__':

    dataset_test = SignTestDataset('./dataset/test', get_transform(train=False))

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=utils.collate_fn)

    prediction_boxes, prediction_masks, prediction_scores = batch_inference(data_loader_test)

    output = []
    for i in tqdm(range(len(prediction_boxes))):
        res = predict_points(prediction_boxes[i], prediction_masks[i], prediction_scores[i], [0.875, 0.875, 0.875])
        for k in res:
            output.append([i+1] + k)
           
    pd.DataFrame(output).to_csv('./dataset/submit.csv', index=False, header=False)
