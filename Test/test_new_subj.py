import os
import time

import nibabel as nib
import numpy as np

import torch
from torch import nn

from Network.FFMNet import MixAttNet as Net

# define the model loading path (set to "latest_train.pth.gz")
ckpt_path = os.path.join('/fileserver/fetal/forWUSTL/fromWUSTL_refined/FetalCPSeg-Programe/Test/ckpt_save/latest_train.pth.gz')

# define the data path (must specify an image file)
data_path = os.path.join("/fileserver/fetal/forWUSTL/fromWUSTL_refined/Input/FCB008_s1/image.nii.gz")

# define the segmentation save path (specify a segmentation file output)
save_path = os.path.join("/fileserver/fetal/forWUSTL/fromWUSTL_refined/try2Haoran")

patch_size = 64
# patch sampling spacing
spacing = 4

# here we define the network
net = Net().cuda()
# I trained this network with DataParallel mode, so we need open this mode in the test phase as well.
net = nn.DataParallel(net)

# load the trained model
print("[*] Loading trained model...".format(ckpt_path))
net.load_state_dict(torch.load(ckpt_path))

net.eval()

print("[*] Loading data...".format(data_path))
data = nib.load(data_path).get_fdata().astype(np.float32)
data = (data - np.min(data))/(np.max(data) - np.min(data))
print("[*] Data loaded with shape {}.".format(data.shape))

w, h, d = data.shape

pre_count = np.zeros_like(data, dtype=np.float32)
predict = np.zeros_like(data, dtype=np.float32)

# here we generate the patch sampling coordinate index of x,y,z
x_list = np.squeeze(np.concatenate((np.arange(0, w - patch_size, patch_size // spacing)[:, np.newaxis],
                                    np.array([w - patch_size])[:, np.newaxis])).astype(np.int))
y_list = np.squeeze(np.concatenate((np.arange(0, h - patch_size, patch_size // spacing)[:, np.newaxis],
                                    np.array([h - patch_size])[:, np.newaxis])).astype(np.int))
z_list = np.squeeze(np.concatenate((np.arange(0, d - patch_size, patch_size // spacing)[:, np.newaxis],
                                    np.array([d - patch_size])[:, np.newaxis])).astype(np.int))
print("[*] Starting test...")
for x in x_list:
    for y in y_list:
        for z in z_list:
            image_patch = data[x:x + patch_size, y:y + patch_size, z:z + patch_size]
            patch_tensor = torch.from_numpy(image_patch[np.newaxis, np.newaxis, ...]).cuda()
            predict[x:x + patch_size, y:y + patch_size, z:z + patch_size] += net(patch_tensor).squeeze().cpu().data.numpy()
            pre_count[x:x + patch_size, y:y + patch_size, z:z + patch_size] += 1

# get the final prediction
predict /= pre_count

predict = np.squeeze(predict)
predict[predict > 0.5] = 1
predict[predict < 0.5] = 0
predict_nii = nib.Nifti1Image(predict, affine=np.eye(4))
nib.save(predict_nii, save_path)
print("[*] Finished.")
