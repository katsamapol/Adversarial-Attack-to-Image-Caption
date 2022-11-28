import torch
import torch.backends.cudnn as cudnn
from datasets import *
from utils import *


# Working path
path = '/scratch/ps4534/ml/data/'

coco2014_folder = path + 'processed_HDF5/COCO2014/'
coco2014_name = 'coco2014_5_cap_per_img_5_min_word_freq'

flickr8k_folder = path + 'processed_HDF5/Flickr8k/' 
flickr8k_name = 'flickr8k_5_cap_per_img_5_min_word_freq'

flickr30k_folder = path + 'processed_HDF5/Flickr30k/' 
flickr30k_name = 'flickr30k_5_cap_per_img_5_min_word_freq'

data_folder = flickr8k_folder
data_name = flickr8k_name

# Cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
print(f"Device: {device}")
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

# Generate mean and std
# train_loader = torch.utils.data.DataLoader(
#     CaptionDataset(data_folder, data_name, 'TRAIN'),
#     batch_size=32, shuffle=True, num_workers=1, pin_memory=True)
# val_loader = torch.utils.data.DataLoader(
#     CaptionDataset(data_folder, data_name, 'VAL'),
#     batch_size=32, shuffle=True, num_workers=1, pin_memory=True)
test_loader = torch.utils.data.DataLoader(
    CaptionDataset(data_folder, data_name, 'TEST'),
    batch_size=128, shuffle=True, num_workers=1, pin_memory=True)

# train_mean, train_std = batch_mean_and_std(train_loader, device)
# print(f"Train: {train_mean}, {train_std}")
# val_mean, val_std = batch_mean_and_std(val_loader, device)
# print(f"Val: {val_mean}, {val_std}")
test_mean, test_std = batch_mean_and_std(test_loader, device)
print(f"Test: {test_mean}, {test_std}")

# Flickr8K
# Train: tensor([0.4580, 0.4464, 0.4032]), tensor([0.2704, 0.2630, 0.2776])
# Val: tensor([0.4573, 0.4448, 0.4060]), tensor([0.2713, 0.2647, 0.2799])
# Test: tensor([0.4615, 0.4501, 0.4107]), tensor([0.2699, 0.2637, 0.2809])

# COCO
# Train: tensor([0.4702, 0.4470, 0.4078]), tensor([0.2701, 0.2655, 0.2809])
# Val: tensor([0.4688, 0.4453, 0.4054]), tensor([0.2691, 0.2642, 0.2802])
# Test: tensor([0.4671, 0.4444, 0.4057]), tensor([0.2705, 0.2663, 0.2814])

# ImageNet ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])