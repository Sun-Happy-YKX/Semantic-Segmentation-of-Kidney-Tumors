import numpy as np
import cv2
from PIL import Image
import os
import sys
sys.path.append(os.pardir)
from starter_code.utils import load_case

# 1. 图片集中存在肾脏和存在肿瘤的图片比例是多少
total_kidney_num, total_tumor_num, total_seg_img = 0.0, 0.0, 0.0
# 2. 图片集中出现肾脏和出现肿瘤的图片中他们所占像素百分比值
kidney_ratio_list, tumor_ratio_list = [], []
# 3、 图片标准差和均值计算

f = open ('data_statistic.txt','a')

for i in range(0, 210):
    volume, segmentation = load_case(i)
    seg_data = np.array(segmentation.get_fdata())
    for j in range(seg_data.shape[0]):
        total_seg_img += 1
        area = seg_data[j].shape[0] * seg_data[j].shape[1]
        kidney_idx = seg_data[j] == 1
        kidney_ratio = len(seg_data[j][kidney_idx]) * 100 / area  #  百分比单位
        if kidney_ratio > 0:
            kidney_ratio_list.append(kidney_ratio)
            total_kidney_num += 1
        tumor_idx = seg_data[j] == 2
        tumor_ratio = len(seg_data[j][tumor_idx]) * 100 / area #  百分比单位
        if tumor_ratio > 0:
            tumor_ratio_list.append(tumor_ratio)
            total_tumor_num += 1

print('肾脏图片存在图片数:{}，总图片数:{}，肾脏图片存在百分率:{:.2f}%'.format(int(total_kidney_num), int(total_seg_img), total_kidney_num * 100 / total_seg_img), file=f)
print('肿瘤图片存在图片数:{}，总图片数:{}，肿瘤图片存在百分率:{:.2f}%'.format(int(total_tumor_num), int(total_seg_img), total_tumor_num * 100 / total_seg_img), file=f)
print('肾脏图片所占像素平均百分比:{:.2f}%'.format(sum(kidney_ratio_list)/len(kidney_ratio_list)), file=f)
print('肿瘤图片所占像素平均百分比:{:.2f}%'.format(sum(tumor_ratio_list)/len(tumor_ratio_list)), file=f)
