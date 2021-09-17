import re
import os
import cv2
import json
import torch
import pickle
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt



img_aicup = sorted(os.listdir(os.path.join('./data/aicup', "TrainDataset/img")))
img_rects = sorted(os.listdir(os.path.join('./data/aicup', "ReCTS/img")))[1:]

gt_aicup = sorted(os.listdir(os.path.join('./data/aicup', "TrainDataset/json")))
gt_rects = sorted(os.listdir(os.path.join('./data/aicup', "ReCTS/gt_unicode")))


dev_index_aicup = set(np.random.choice(list(range(1, len(img_aicup)+1)), 400, replace=False))
dev_inde_rects = set(np.random.choice(list(range(1, len(img_rects)+1)), 200, replace=False))


n = len(img_aicup)
for i in tqdm(range(1, n+1)):
    if i in dev_index_aicup:
        os.system('mv ./dataset/train/img/img_{}.jpg ./dataset/dev/img/img_{}.jpg'.format(i, i))

n = len(img_rects)
for i in tqdm(range(1, n+1)):
    if i in dev_inde_rects:
        os.system('mv ./dataset/train/img/train_ReCTS_{0:06d}.jpg ./dataset/dev/img/train_ReCTS_{0:06d}.jpg'.format(i, i))


def gen_info_rects(img, json_file):

    with open(json_file, 'r') as file:
        data = json.load(file)
    file.close()
    num_objs = len(data['lines'])
    num_lines = len(data['lines'])
    img_cv = np.array(img)[:, :, ::-1]
    masks, boxes, areas = np.zeros((num_objs, img_cv.shape[0], img_cv.shape[1])), [], []
    height, width = img_cv.shape[0], img_cv.shape[1]
    jj = 0
    for shape in data['lines']:
        points = shape['points']
        points = [(points[0], points[1]), (points[2], points[3]), (points[4], points[5]), (points[6], points[7])]
        loc, area = to_rec(points)
        if check_valid_box(loc, height, width):
            masks = masks[:-1, :, :]
            num_objs -= 1
            num_lines -= 1
            continue
        masks[jj, :, :] = gen_mask(img_cv, points)
        boxes.append(loc)
        areas.append(area)
        jj += 1
        

    masks = np.array(masks).astype(np.uint8)
    assert len(masks) == len(boxes) == len(areas) == len(areas) == num_objs
    return masks, boxes, areas, num_objs, num_lines


def gen_info_aicup(img, json_file):

    with open(json_file, 'r') as file:
        data = json.load(file)
    file.close()
    num_objs = len(data['shapes'])
    num_lines = len(data['shapes'])
    img_cv = np.array(img)[:, :, ::-1]
    masks, boxes, areas = np.zeros((num_objs, img_cv.shape[0], img_cv.shape[1])), [], []
    height, width = img_cv.shape[0], img_cv.shape[1]
    jj = 0
    for shape in data['shapes']:
        points = shape['points']
        loc, area = to_rec(points)
        if check_valid_box(loc, height, width):
            masks = masks[:-1, :, :]
            num_objs -= 1
            num_lines -= 1
            continue
        masks[jj, :, :] = gen_mask(img_cv, points)
        boxes.append(loc)
        areas.append(area)
        jj += 1

    masks = np.array(masks).astype(np.uint8)
    assert len(masks) == len(boxes) == len(areas) == num_objs == num_lines
    return masks, boxes, areas, num_objs, num_lines


def to_rec(points):
    xmin = round(min(points[0][0], points[3][0]))
    ymin = round(min(points[0][1], points[1][1]))
    xmax = round(max(points[1][0], points[2][0]))
    ymax = round(max(points[2][1], points[3][1]))
    return [xmin, ymin, xmax, ymax], (xmax-xmin)*(ymax-ymin)


def gen_mask(img, points):
    masked_image = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    mask.fill(255)
    roi_corners = np.array([[(x[0], x[1]) for x in points]], dtype=np.int32)
    cv2.fillPoly(mask, roi_corners, 0)
    masked_image = cv2.bitwise_or(masked_image, mask)
    masked_image[masked_image == 0] = 1
    masked_image[masked_image == 255] = 0
    return masked_image


def check_valid_box(loc, height, width):
    xmin, ymin, xmax, ymax = loc
    if xmin < 0 or ymin < 0 or xmax < 0 or ymax < 0:
        return 1
    elif xmax > width or ymax > height:
        return 1
    elif xmax <= xmin or ymax <= ymin:
        return 1
    return 0


def generate_mask_and_gt(folder):
    dir_path = './dataset/{}/img'.format(folder)
    index_map, index = {}, 0
    for img_name in tqdm(sorted(os.listdir(dir_path), reverse=True)):

        img = Image.open(os.path.join(dir_path, img_name)).convert("RGB")
        img_size = np.array(img)
        if img_name[:3] == 'img':
            idx = re.findall(r'\_(.*)\.', img_name)[0]
            json_file = './data/aicup/TrainDataset/json/'+'img_{}.json'.format(idx)
            masks, boxes, areas, num_objs, num_lines = gen_info_aicup(img, json_file)
        else:
            idx = int(re.findall(r'ReCTS_(.*)\.', img_name)[0])
            json_file = './data/aicup/ReCTS/gt_unicode/'+'train_ReCTS_{0:06d}.json'.format(idx)
            masks, boxes, areas, num_objs, num_lines = gen_info_rects(img, json_file)

        gt_info = {
            'img_name': img_name,
            'boxes': boxes,
            'areas': areas,
            'num_objs': num_objs,
            'num_lines': num_lines,
            'size': [img_size.shape[0], img_size.shape[1], img_size.shape[2]]
        }
        
        if num_objs:
            index_map[index] = img_name.split('.')[0]
            index += 1
            with open(os.path.join('./dataset/{}/gt'.format(folder), img_name.split('.')[0]+'.pickle'), 'wb') as output_file:
                pickle.dump(gt_info, output_file, protocol=pickle.HIGHEST_PROTOCOL)
            output_file.close()

            masks_img = np.reshape(masks, (masks.shape[0]*masks.shape[1], masks.shape[2]))
            cv2.imwrite(os.path.join('./dataset/{}/mask'.format(folder), img_name.split('.')[0]+'.png'), masks_img)
            
    with open('./dataset/{}/index_map.pickle'.format(folder), 'wb') as output_file:
        pickle.dump(index_map, output_file, protocol=pickle.HIGHEST_PROTOCOL)
    output_file.close()


if __name__ == '__main__':
    generate_mask_and_gt('train')
    generate_mask_and_gt('dev')