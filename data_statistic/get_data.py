import numpy as np
import cv2
from PIL import Image
import os
import sys
sys.path.append(os.pardir)
from starter_code.utils import load_case
import shutil
# cd /data/yangkaixing/kits19/data_statistic
# nohup python get_data.py > get_data.log 2>&1 &

people_num = 210
divide_num = int(210 * 0.7)
root_train_img_path = '/data/yangkaixing/kits19/Main/data/images/train'
root_train_label_path = '/data/yangkaixing/kits19/Main/data/labels/train'
root_test_img_path = '/data/yangkaixing/kits19/Main/data/images/test'
root_test_label_path = '/data/yangkaixing/kits19/Main/data/labels/test'

shutil.rmtree(root_train_img_path, ignore_errors=True)
shutil.rmtree(root_train_label_path, ignore_errors=True)
shutil.rmtree(root_test_img_path, ignore_errors=True)
shutil.rmtree(root_test_label_path, ignore_errors=True)
os.makedirs(root_train_img_path, exist_ok=True)
os.makedirs(root_train_label_path, exist_ok=True)
os.makedirs(root_test_img_path, exist_ok=True)
os.makedirs(root_test_label_path, exist_ok=True)


train_txt = '/data/yangkaixing/kits19/Main/data/train.txt'
test_txt = '/data/yangkaixing/kits19/Main/data/test.txt'
train_f = open(train_txt, 'w')
test_f = open(test_txt, 'w')

for i in range(0, people_num):

    volume, segmentation = load_case(i)
    seg_data = np.array(segmentation.get_fdata())
    img_data = np.array(volume.get_fdata())

    for j in range(seg_data.shape[0]):

        kidney_idx = seg_data[j] == 1
        kidney_exist = len(seg_data[j][kidney_idx]) > 0

        if kidney_exist > 0: # 我们只把存在肾的数据图片提取出来

            if i < divide_num:

                # train_img_path = os.path.join(root_train_img_path, str(i))
                # train_label_path = os.path.join(root_train_label_path, str(i))
                # os.makedirs(train_img_path, exist_ok=True)
                # os.makedirs(train_label_path, exist_ok=True)
                img_name = os.path.join(root_train_img_path, '{:0>4d}_{:0>4d}.png'.format(i, j))
                label_name = os.path.join(root_train_label_path, '{:0>4d}_{:0>4d}.npy'.format(i, j))
                train_f.write(img_name + ' ' + label_name + '\n')

            else:

                # test_img_path = os.path.join(root_test_img_path, str(i))
                # test_label_path = os.path.join(root_test_label_path, str(i))
                # os.makedirs(test_img_path, exist_ok=True)
                # os.makedirs(test_label_path, exist_ok=True)
                img_name = os.path.join(root_test_img_path, '{:0>4d}_{:0>4d}.png'.format(i, j))
                label_name = os.path.join(root_test_label_path, '{:0>4d}_{:0>4d}.npy'.format(i, j))
                test_f.write(img_name + ' ' + label_name + '\n')

            img = Image.fromarray(img_data[j]).convert('RGB')
            img.save(img_name)

            one_hot_values= np.eye(3)[seg_data[j].astype(np.uint8)]
            label = np.uint8(one_hot_values)
            np.save(label_name, label)
            # label = (seg_data[j] * (255.0 / 2)).astype(np.uint8) 
            # label = Image.fromarray(label)
            # label.save(label_name)

train_f.close()
test_f.close()
