import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Main.utils.dataset import load_data
from tqdm import tqdm
import yaml

def get_mean_std(train_loader):
    # var[X] = E[X**2] - E[X]**2
    channels_sum, channels_sqrd_sum, num_batches = 0, 0, 0

    for data, _ in tqdm(train_loader['train']):
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_sqrd_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_sqrd_sum / num_batches - mean ** 2) ** 0.5

    return mean, std

with open('/data/yangkaixing/kits19/Main/configs/parameter.yaml', 'r', encoding='utf-8') as f:
    param_dict = yaml.load(f, Loader=yaml.FullLoader)
train_loader = load_data(params=param_dict)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mean, std = get_mean_std(train_loader)
print("mean is {}".format(mean.cpu().numpy()))
print("std is {}".format(std.cpu().numpy()))