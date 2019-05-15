import fnmatch
import math
import os
import sys
from operator import itemgetter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from darknet import Darknet

class PatchTransformer(nn.Module):
    """PatchTransformer: transforms batch of patches

    Module providing the functionality necessary to transform a batch of patches, randomly adjusting brightness and
    contrast, adding random amount of noise, and rotating randomly. Resizes patches according to as size based on the
    batch of labels, and pads them to the dimension of an image.

    """

    def __init__(self):
        super(PatchTransformer, self).__init__()
        self.min_contrast = 0.7
        self.max_contrast = 1.3
        self.min_brightness = -0.2
        self.max_brightness = 0.2
        self.noise_factor = 0.15
        self.minangle = -180/180*math.pi
        self.maxangle = 180/180*math.pi

    def forward(self, adv_patch, lab_batch, img_size):
        # Determine size of padding
        pad = (img_size - adv_patch.size(-1))/2

        # Make a batch of patches
        adv_patch = adv_patch.unsqueeze(0).unsqueeze(0)
        adv_batch = adv_patch.expand(lab_batch.size(0), lab_batch.size(1), -1, -1, -1)

        # Contrast, brightness and noise transforms
        batch_size = (lab_batch.size(0),lab_batch.size(1))


        # Create random contrast tensor
        contrast = torch.zeros(batch_size).uniform_(self.min_contrast,self.max_contrast)
        contrast = contrast.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        contrast = contrast.expand(-1,-1, adv_batch.size(-3), adv_batch.size(-2),adv_batch.size(-1))

        # Create random brightness tensor
        brightness = torch.zeros(batch_size).uniform_(self.min_brightness,self.max_brightness)
        brightness = brightness.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        brightness = brightness.expand(-1,-1, adv_batch.size(-3), adv_batch.size(-2),adv_batch.size(-1))

        # Create random noise tensor
        noise = torch.zeros(adv_batch.size()).uniform_(-1,1)*self.noise_factor

        # Apply contrast/brightness/noise, clamp
        adv_batch = adv_batch*contrast + brightness + noise
        adv_batch = torch.clamp(adv_batch, 0, 0.99999)
        
        print('adv_batch.shape', adv_batch.shape)

        '''
        img = adv_batch[0,0,:,:,:]
        img = transforms.ToPILImage()(img)
        img.show()
        '''

        # Where the label class_id is 1 we don't want a patch --> fill mask with zero's
        cls_ids = torch.narrow(lab_batch, 2, 0, 1).float()
        cls_mask = cls_ids.expand(-1, -1, 3)
        cls_mask = cls_mask.unsqueeze(-1)
        cls_mask = cls_mask.expand(-1, -1, -1, adv_batch.size(3))
        cls_mask = cls_mask.unsqueeze(-1)
        cls_mask = cls_mask.expand(-1, -1, -1, -1, adv_batch.size(4))
        msk_batch = torch.ones(cls_mask.size()) - cls_mask

        # Pad patch and mask to image dimensions
        mypad = nn.ConstantPad2d((int(pad + 0.5), int(pad), int(pad + 0.5), int(pad)), 0)
        adv_batch = mypad(adv_batch)
        msk_batch = mypad(msk_batch)

        '''
        img = adv_batch[0, 0, :, :, :]
        img = transforms.ToPILImage()(img)
        img.show()
        '''

        # Rotation and rescaling transforms
        anglesize = (lab_batch.size(0)*lab_batch.size(1))
        angle = torch.empty(anglesize).uniform_(self.minangle, self.maxangle)

        # Resizes and rotates
        current_patch_size = adv_patch.size(-1)
        lab_batch_scaled = torch.zeros(lab_batch.size())
        lab_batch_scaled[:, :, 1] = lab_batch[:, :, 1]*img_size
        lab_batch_scaled[:, :, 2] = lab_batch[:, :, 2]*img_size
        lab_batch_scaled[:, :, 3] = lab_batch[:, :, 3]*img_size
        lab_batch_scaled[:, :, 4] = lab_batch[:, :, 4]*img_size
        target_size = torch.sqrt((lab_batch_scaled[:, :, 3].mul(0.2)) ** 2 + (lab_batch_scaled[:, :, 4].mul(0.2)) ** 2)
        target_x = lab_batch[:, :, 1].view(np.prod(batch_size))
        target_y = lab_batch[:, :, 2].view(np.prod(batch_size))
        scale = target_size/current_patch_size
        scale = scale.view(anglesize)

        s = adv_batch.size()
        adv_batch = adv_batch.view(s[0]*s[1],s[2],s[3],s[4])
        msk_batch = msk_batch.view(s[0]*s[1],s[2],s[3],s[4])

        # Theta = rotation,rescale matrix
        theta = torch.zeros(anglesize, 2, 3)
        theta[:, 0, 0] = torch.cos(angle)//scale
        theta[:, 0, 1] = torch.sin(angle)//scale
        theta[:, 0, 2] = 0#target_x
        theta[:, 1, 0] = -torch.sin(angle)//scale
        theta[:, 1, 1] = torch.cos(angle)//scale
        theta[:, 1, 2] = 0#target_y
        grid = F.affine_grid(theta, adv_batch.shape)
        adv_batch_t = F.grid_sample(adv_batch, grid)
        msk_batch_t = F.grid_sample(msk_batch, grid)

        # Theta2 = translation matrix
        theta2 = torch.zeros(anglesize, 2, 3)
        theta2[:, 0, 0] = 1
        theta2[:, 0, 1] = 0
        theta2[:, 0, 2] = (-target_x+0.5)*2
        theta2[:, 1, 0] = 0
        theta2[:, 1, 1] = 1
        theta2[:, 1, 2] = (-target_y+0.5)*2
        grid2 = F.affine_grid(theta2, adv_batch.shape)
        adv_batch_t = F.grid_sample(adv_batch_t, grid2)
        adv_batch_t = adv_batch_t.view(s[0], s[1], s[2], s[3], s[4])
        msk_batch_t = F.grid_sample(msk_batch_t, grid2)
        msk_batch_t = msk_batch_t.view(s[0], s[1], s[2], s[3], s[4])
        adv_batch_t = torch.clamp(adv_batch_t, 0, 1)
        '''
        img = adv_batch_t[0, 0, :, :, :]
        img = transforms.ToPILImage()(img)
        img.show()
        '''

        return adv_batch_t*msk_batch_t

class PatchApplier(nn.Module):
    """PatchApplier: applies adversarial patches to images.

    Module providing the functionality necessary to apply a patch to all detections in all images in the batch.

    """

    def __init__(self):
        super(PatchApplier, self).__init__()

    def forward(self, img_batch, adv_batch):
        print('img_batch.shape',img_batch.shape)
        print('adv_batch.shape',adv_batch.shape)
        advs = torch.unbind(adv_batch, 1)
        for adv in advs:
            img_batch = torch.where((adv==0),img_batch,adv)
        return img_batch

class InriaDataset(Dataset):
    """InriaDataset: representation of the INRIA person dataset.

    Internal representation of the commonly used INRIA person dataset.
    Available at: http://pascal.inrialpes.fr/data/human/

    Attributes:
        len: An integer number of elements in the
        img_dir: Directory containing the images of the INRIA dataset.
        lab_dir: Directory containing the labels of the INRIA dataset.
        img_names: List of all image file names in img_dir.
        shuffle: Whether or not to shuffle the dataset.

    """

    def __init__(self, img_dir, lab_dir, shuffle=True):
        n_png_images = len(fnmatch.filter(os.listdir(img_dir), '*.png'))
        n_jpg_images = len(fnmatch.filter(os.listdir(img_dir), '*.jpg'))
        n_images = n_png_images + n_jpg_images
        n_labels = len(fnmatch.filter(os.listdir(lab_dir), '*.txt'))
        assert n_images == n_labels, "Number of images and number of labels don't match"
        self.len = n_images
        self.img_dir = img_dir
        self.lab_dir = lab_dir
        self.img_names = fnmatch.filter(os.listdir(img_dir), '*.png') + fnmatch.filter(os.listdir(img_dir), '*.jpg')
        self.shuffle = shuffle
        self.img_paths = []
        for img_name in self.img_names:
            self.img_paths.append(os.path.join(self.img_dir, img_name))
        sizes = [Image.open(f, 'r').size for f in self.img_paths]
        self.max_im_width = max(sizes,key=itemgetter(0))[0]
        self.max_im_height = max(sizes,key=itemgetter(1))[1]
        self.lab_paths = []
        for img_name in self.img_names:
            lab_path = os.path.join(self.lab_dir, img_name).replace('.jpg', '.txt').replace('.png', '.txt')
            self.lab_paths.append(lab_path)
        self.max_n_labels = 0
        for lab_file in self.lab_paths:
            with open(lab_file) as f:
                for i, l in enumerate(f):
                    pass
            line_count =  i + 1
            self.max_n_labels = max(line_count,self.max_n_labels)
        #print(self.max_n_labels)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        assert idx <= len(self), 'index range error'
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        lab_path = os.path.join(self.lab_dir, self.img_names[idx]).replace('.jpg', '.txt').replace('.png', '.txt')
        image = Image.open(img_path).convert('RGB')
        label = torch.from_numpy(np.loadtxt(lab_path))
        if label.dim() == 1:
            label = label.unsqueeze(0)
        image, label = self.pad_and_scale(image, label)
        transform = transforms.ToTensor()
        image = transform(image)
        label = self.pad_lab(label)
        return image, label

    def pad_and_scale(self, img, lab):
        """

        Args:
            img:

        Returns:

        """
        w,h = img.size
        if w==h:
            padded_img = img
        else:
            dim_to_pad = 1 if w<h else 2
            if dim_to_pad == 1:
                padding = (h - w) / 2
                padded_img = Image.new('RGB', (h,h), color=(127,127,127))
                padded_img.paste(img, (int(padding), 0))
                lab[:, [1]] = (lab[:, [1]] * w + padding) / h
                lab[:, [3]] = (lab[:, [3]] * w / h)
            else:
                padding = (w - h) / 2
                padded_img = Image.new('RGB', (w, w), color=(127,127,127))
                padded_img.paste(img, (0, int(padding)))
                lab[:, [2]] = (lab[:, [2]] * h + padding) / w
                lab[:, [4]] = (lab[:, [4]] * h  / w)
        max_im_dim = max(self.max_im_width, self.max_im_height)
        padded_img = padded_img.resize((600,600))     #choose here
        return padded_img, lab

    def pad_lab(self, lab):
        pad_size = self.max_n_labels - lab.shape[0]
        if(pad_size>0):
            padded_lab = F.pad(lab, (0, 0, 0, pad_size), value=1)
        else:
            padded_lab = lab
        return padded_lab


if __name__ == '__main__':
    if len(sys.argv) == 3:
        img_dir = sys.argv[1]
        lab_dir = sys.argv[2]

    else:
        print('Usage: ')
        print('  python load_data.py img_dir lab_dir')
        sys.exit()

    test_loader = torch.utils.data.DataLoader(InriaDataset(img_dir, lab_dir, shuffle=True),
                                              batch_size=5,
                                              shuffle=True)

    cfgfile = "cfg/yolov2.cfg"
    weightsfile = "weights/yolov2.weights"

    darknet_model = Darknet(cfgfile)
    darknet_model.load_weights(weightsfile)
    patchapplier = PatchApplier()
    patchtransformer = PatchTransformer()

    #use_cuda = True
    #if use_cuda:
    #    model = model.cuda()

    for i_batch, (img_batch, lab_batch) in enumerate(test_loader):
        adv_patch = Image.open('data/horse.jpg').convert('RGB')
        adv_patch = adv_patch.resize((400, 400))
        transform = transforms.ToTensor()
        adv_patch = transform(adv_patch)
        img_size = img_batch.size(-1)
        adv_batch_t = patchtransformer.forward(adv_patch, lab_batch, img_size)
        img_batch = patchapplier.forward(img_batch, adv_batch_t)
        #img = img_batch.squeeze(0)
        #img = transforms.ToPILImage()(img)
        #img.show()
