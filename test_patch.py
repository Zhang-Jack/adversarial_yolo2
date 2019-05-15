import sys
import time
import os
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, ImageDraw
from utils import *
from darknet import *
from load_data import PatchTransformer, PatchApplier, InriaDataset
import json


if __name__ == '__main__':
    print("Setting everything up")
    imgdir = "inria/Test/pos"
    cfgfile = "cfg/yolo.cfg"
    weightfile = "weights/yolo.weights"
    patchfile = "saved_patches/patch11.jpg"
    patchfile = "/home/wvr/Pictures/individualImage_upper_body.png"
    #patchfile = "/home/wvr/Pictures/class_only.png"
    #patchfile = "/home/wvr/Pictures/class_transfer.png"
    savedir = "testing"

    darknet_model = Darknet(cfgfile)
    darknet_model.load_weights(weightfile)
    darknet_model = darknet_model.eval().cuda()
    patch_applier = PatchApplier().cuda()
    patch_transformer = PatchTransformer().cuda()

    batch_size = 1
    max_lab = 14
    img_size = darknet_model.height

    patch_size = 300

    patch_img = Image.open(patchfile).convert('RGB')
    tf = transforms.Resize((patch_size,patch_size))
    patch_img = tf(patch_img)
    tf = transforms.ToTensor()
    adv_patch_cpu = tf(patch_img)
    adv_patch = adv_patch_cpu.cuda()

    clean_results = []
    noise_results = []
    patch_results = []
    
    print("Done")
    #Loop over cleane beelden
    for imgfile in os.listdir(imgdir):
        print("new image")
        if imgfile.endswith('.jpg') or imgfile.endswith('.png'):
            name = os.path.splitext(imgfile)[0]    #image name w/o extension
            txtname = name + '.txt'
            txtpath = os.path.abspath(os.path.join(savedir, 'clean/', 'yolo-labels/', txtname))
            # open beeld en pas aan naar yolo input size
            imgfile = os.path.abspath(os.path.join(imgdir, imgfile))
            img = Image.open(imgfile).convert('RGB')
            w,h = img.size
            if w==h:
                padded_img = img
            else:
                dim_to_pad = 1 if w<h else 2
                if dim_to_pad == 1:
                    padding = (h - w) / 2
                    padded_img = Image.new('RGB', (h,h), color=(127,127,127))
                    padded_img.paste(img, (int(padding), 0))
                else:
                    padding = (w - h) / 2
                    padded_img = Image.new('RGB', (w, w), color=(127,127,127))
                    padded_img.paste(img, (0, int(padding)))
            resize = transforms.Resize((img_size,img_size))
            padded_img = resize(padded_img)
            cleanname = name + ".png"
            #sla dit beeld op
            padded_img.save(os.path.join(savedir, 'clean/', cleanname))
            
            #genereer een label file voor het gepadde beeld
            boxes = do_detect(darknet_model, padded_img, 0.4, 0.4, True)
            boxes = nms(boxes, 0.4)
            textfile = open(txtpath,'w+')
            for box in boxes:
                cls_id = box[6]
                if(cls_id == 0):   #if person
                    x_center = box[0]
                    y_center = box[1]
                    width = box[2]
                    height = box[3]
                    textfile.write(f'{cls_id} {x_center} {y_center} {width} {height}\n')
                    clean_results.append({'image_id': name, 'bbox': [x_center.item() - width.item() / 2,
                                                                     y_center.item() - height.item() / 2,
                                                                     width.item(),
                                                                     height.item()],
                                          'score': box[4].item(),
                                          'category_id': 1})
            textfile.close()

            #lees deze labelfile terug in als tensor            
            if os.path.getsize(txtpath):       #check to see if label file contains data. 
                label = np.loadtxt(txtpath)
            else:
                label = np.ones([5])
            label = torch.from_numpy(label).float()
            if label.dim() == 1:
                label = label.unsqueeze(0)

            
            transform = transforms.ToTensor()
            padded_img = transform(padded_img).cuda()
            img_fake_batch = padded_img.unsqueeze(0)
            lab_fake_batch = label.unsqueeze(0).cuda()
            
            #transformeer patch en voeg hem toe aan beeld
            adv_batch_t = patch_transformer(adv_patch, lab_fake_batch, img_size, do_rotate=True, rand_loc=False)
            p_img_batch = patch_applier(img_fake_batch, adv_batch_t)
            p_img = p_img_batch.squeeze(0)
            p_img_pil = transforms.ToPILImage('RGB')(p_img.cpu())
            properpatchedname = name + "_p.png"
            p_img_pil.save(os.path.join(savedir, 'proper_patched/', properpatchedname))
            
            #genereer een label file voor het beeld met sticker
            txtname = properpatchedname.replace('.png', '.txt')
            txtpath = os.path.abspath(os.path.join(savedir, 'proper_patched/', 'yolo-labels/', txtname))
            boxes = do_detect(darknet_model, p_img_pil, 0.01, 0.4, True)
            boxes = nms(boxes, 0.4)
            textfile = open(txtpath,'w+')
            for box in boxes:
                cls_id = box[6]
                if(cls_id == 0):   #if person
                    x_center = box[0]
                    y_center = box[1]
                    width = box[2]
                    height = box[3]
                    textfile.write(f'{cls_id} {x_center} {y_center} {width} {height}\n')
                    patch_results.append({'image_id': name, 'bbox': [x_center.item() - width.item() / 2, y_center.item() - height.item() / 2, width.item(), height.item()], 'score': box[4].item(), 'category_id': 1})
            textfile.close()

            #maak een random patch, transformeer hem en voeg hem toe aan beeld
            random_patch = torch.rand(adv_patch_cpu.size()).cuda()
            adv_batch_t = patch_transformer(random_patch, lab_fake_batch, img_size, do_rotate=True, rand_loc=False)
            p_img_batch = patch_applier(img_fake_batch, adv_batch_t)
            p_img = p_img_batch.squeeze(0)
            p_img_pil = transforms.ToPILImage('RGB')(p_img.cpu())
            properpatchedname = name + "_rdp.png"
            p_img_pil.save(os.path.join(savedir, 'random_patched/', properpatchedname))
            
            #genereer een label file voor het beeld met random patch
            txtname = properpatchedname.replace('.png', '.txt')
            txtpath = os.path.abspath(os.path.join(savedir, 'random_patched/', 'yolo-labels/', txtname))
            boxes = do_detect(darknet_model, p_img_pil, 0.01, 0.4, True)
            boxes = nms(boxes, 0.4)
            textfile = open(txtpath,'w+')
            for box in boxes:
                cls_id = box[6]
                if(cls_id == 0):   #if person
                    x_center = box[0]
                    y_center = box[1]
                    width = box[2]
                    height = box[3]
                    textfile.write(f'{cls_id} {x_center} {y_center} {width} {height}\n')
                    noise_results.append({'image_id': name, 'bbox': [x_center.item() - width.item() / 2, y_center.item() - height.item() / 2, width.item(), height.item()], 'score': box[4].item(), 'category_id': 1})
            textfile.close()

    with open('clean_results.json', 'w') as fp:
        json.dump(clean_results, fp)
    with open('noise_results.json', 'w') as fp:
        json.dump(noise_results, fp)
    with open('patch_results.json', 'w') as fp:
        json.dump(patch_results, fp)
            

