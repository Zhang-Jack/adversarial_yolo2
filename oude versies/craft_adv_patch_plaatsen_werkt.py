import sys
import time
import os
import numpy as np
import random
import cv2
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from darknet import Darknet
from utils import read_truths
import torchvision.transforms.functional as tvfunc


def get_max_probability(output, cls_id, num_classes):
    '''
    Haalt uit de output van YOLO de maximum class probability over het hele beeld 
    voor een gegeven class.
    '''
    # get values neccesary for transformation
    if output.dim() == 3:
        output = output.unsqueeze(0)
    batch = output.size(0)
    assert(output.size(1) == (5+num_classes)*5)
    h = output.size(2)
    w = output.size(3)

    # transform the output tensor from [1, 425, 19, 19] to [80, 1805]
    output = output.view(batch*5, 5+num_classes, h*w)   #[5, 85, 361]
    output = output.transpose(0,1).contiguous() #[85, 5, 361]
    output = output.view(5+num_classes, batch*5*h*w)    #[85, 1805]
    output = output[5:5+num_classes]    #[80, 1805]

    # perform softmax to normalize probabilities for object classes to [0,1]
    cls_confs = torch.nn.Softmax(dim=0)(Variable(output)).data

    # we only care for probabilities of the class of interest (person)
    probs_for_class = cls_confs[cls_id,:]

    # find the max probability for person
    max_prob, max_ind = probs_for_class.max(0)

    return max_prob

def get_printability_array(printability_file, height, width):
    '''
    Leest het bestand met de printbare RGB-waarden in 
    en geeft een array terug van de vorm (aantal_triplets, aantal_kleurkanalen, height, width).
    Maakt in essentie voor elke triplet een beeld met dezelfde dimensies als de afbeelding, maar 
    gevuld met de kleur gedefinieerd door de triplet. 
    '''
    printability_list = []
    printability_array = []

    # read in printability triplets and put them in a list
    with open(printability_file) as f:
        for line in f:
            printability_list.append(line.split(","))
    
    for printability_triplet in printability_list:
        printability_imgs = []
        red, green, blue = printability_triplet
        printability_imgs.append(np.full((height, width),red))
        printability_imgs.append(np.full((height, width),green))
        printability_imgs.append(np.full((height, width),blue))
        printability_array.append(printability_imgs)
        #print('tile',np.tile(printability_triplet, (height, width, 1)).shape)

    printability_array = np.asarray(printability_array)
    printability_array = np.float32(list(printability_array))
    #print('pri',printability_array.shape)
    return torch.from_numpy(printability_array).double()

#    mask = load_norm_mask()
#
#   # mask out triplets outside the mask
#    for t in p:
#        for x in xrange(FLAGS.img_cols):
#            for y in xrange(FLAGS.img_rows):
#                if np.all(mask[x][y] == 0.0):
#                    t[x][y] = [0.0,0.0,0.0]

def calc_nps(adv_patch, printability_array):
    '''
    Functie die de non-printability score berekent voor de aangeleverde adversarial patch en 
    de printability_array 
    '''
    rombom = adv_patch - printability_array
    rambam = torch.sum(rombom**2,1)
    print('rombom')
    print(rombom)
    print(rambam)
    print(rombom.shape)
    # bereken de afstand tussen kleuren in adv_patch en printability_array per pixel 
    color_dist_per_printable_color = torch.sqrt(torch.sum((adv_patch - printability_array)**2,1))

    # product van de afstanden over alle kleuren in printability_array
    prod_of_color_distances = torch.prod(color_dist_per_printable_color,0)

    # som over het volledige beeld
    nps_score = torch.sum(prod_of_color_distances)

    return nps_score

def total_variation(adv_patch):
    '''
    Berekent de total variation van de gegeven adversarial patch, en normaliseert deze 
    ten opzichte van het aantal pixels in de patch.
    '''
    numel = torch.numel(adv_patch)/3
    tv  = torch.sum(torch.abs(adv_patch[:, :, 1:] - adv_patch[:, :, :-1])) 
    tv += torch.sum(torch.abs(adv_patch[:, 1:, :] - adv_patch[:, :-1, :]))
    tv = tv/numel
    return tv

def read_and_size_image(imgfile, width=None, height=None):
    '''
    Leest het beeld met path imgfile in als PIL Image en transformeert het 
    naar een tensor met waarden in [0,1]
    '''
    img = Image.open(imgfile).convert('RGB')
    if width and height:
        img = img.resize((width, height))
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
        img = img.view(height, width, 3).transpose(0,1).transpose(0,2).contiguous()
        img = img.view(1, 3, height, width)
    else:
        width, height = img.size
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
        img = img.view(height, width, 3).transpose(0,1).transpose(0,2).contiguous()
        img = img.view(1, 3, height, width)
    img = img.float().div(255.0)
    
    return img

def apply_patch(img, adv_patch, truths):
    '''
    Plaatst de adversarial patch in adv_patch (Tensor) op het beeld img (Tensor) 
    op basis van de locatie van de bounding boxes in bbox. 
    adv_patch:  tensor met dim [channels][adv_height][adv_width]
    img:        tensor met dim [channels][img_height][img_width]
    truths:     array van bbox-locaties zoals gelezen uit .txt-file
    '''
    #verwijder thruths die niet van class 0 zijn
    print(truths.shape)
    truths = truths[truths[:,0]==0]

    #haal de kolom met classes weg
    truths = truths[:,1:]

    #converteer thruths naar pixellocaties op basis van de dimensies van img
    truths[:,0] *= img.shape[3]
    truths[:,1] *= img.shape[2]
    truths[:,2] *= img.shape[3]
    truths[:,3] *= img.shape[2]
    print('img.shape',img.shape)
    print('truths[:,2]:',truths[:,2])

    #transformeer adv_patches en plaats ze op de afbeelding
    patched_img = img
    for truth in truths:
        print(truth)
        #bereken de dimensie van adv_patch a.d.h.v. de thruth
        patch_dim = truth[2].mul(0.5).int()
        print('patch_dim',patch_dim)
        #resize de patch naar deze dimensie
        resized_patch = F.interpolate(adv_patch, size=(patch_dim,patch_dim), mode='bilinear', align_corners=True)
        print('resized_patch.shape',resized_patch.shape)
        #bepaal de padding van de patch
        int_truth = truth.int()
        print('int_truth',int_truth)
        lpad = int_truth[0]-(patch_dim/2).int()
        rpad = img.shape[3]-lpad-patch_dim
        tpad = int_truth[1]-(patch_dim/2).int()
        bpad = img.shape[2]-tpad-patch_dim
        print('lpad:',lpad,'rpad:',rpad,'tpad:',tpad,'bpad:',bpad)
        mypad = nn.ConstantPad2d((lpad,rpad,tpad,bpad), 2)
        padded_patch = mypad(resized_patch)
        print('padded_patch',padded_patch)
        print('patched_img.type()',patched_img.type())
        patched_img[padded_patch!=2] = padded_patch[padded_patch!=2]
    zien = tvfunc.to_pil_image(patched_img.squeeze(0))
    zien.show()

    #patch kopieren per detectie in truth
    #n_truths = truths.shape[0]
    #expanded_patch = adv_patch.repeat(n_truths,1,1,1)
    #print('expanded_patch.shape',expanded_patch.shape)

    #adv_patch resizen tot de gewenste grootte
    #resized_patches = F.interpolate(expanded_patch, size=(patch_dim,patch_dim), mode='bilinear', align_corners=True)

    #patch padden met nullen

    #patches aan inputbeeld toevoegen door gebruik van where
    #img = img
    #adv_patch = adv_patch
    #for thruth in thruths:
    #    if thruth[0] == 0:
    #        resized_patch = F.interpolate(adv_patch, size=(25,20), mode='bilinear')
    

if __name__ == '__main__':
    if len(sys.argv) == 4:
        cfgfile = sys.argv[1]
        weightfile = sys.argv[2]
        imgfile = sys.argv[3]
        #detect(cfgfile, weightfile, imgfile)

    else:
        print('Usage: ')
        print('  python craft_adv.py cfgfile weightfile imgfile')
        sys.exit()
'''
    # load the darknet model and weights
    darknet_model = Darknet(cfgfile)
    darknet_model.print_network()
    darknet_model.load_weights(weightfile)
    
    # read in the label names associated with the darknet model    
    if darknet_model.num_classes == 20:
        namesfile = 'data/voc.names'
    elif darknet_model.num_classes == 80:
        namesfile = 'data/coco.names'
    else:
        namesfile = 'data/names'
'''
imgfile2 = 'inria/Train/pos/crop001002.png'
img = read_and_size_image(imgfile)
img2 = read_and_size_image(imgfile2)
'''
    sized = img.resize((darknet_model.width, darknet_model.height))
    
    # move the darknet model to the GPU
    darknet_model = darknet_model.cuda()
    sized.cuda()
    sized = torch.autograd.Variable(sized)
    output = model.forward(darknet_model,sized)
    get_max_probability(output, 0,80)
'''
print(img.shape)
img_interp = F.interpolate(img, size=(200,200), mode='bilinear', align_corners=True)

#zien = tvfunc.to_pil_image(img_interp.squeeze(0))
#zien.show()

printability_file = 'non_printability/30values.txt'
img_height = 500
img_width = 500
printability_array = get_printability_array(printability_file, img_height, img_width)
good_patch = torch.from_numpy(np.tile([0.7098,0.32157,0.2],(img_height,img_width,1))).float()
good_patch = good_patch.view(img_height, img_width, 3).transpose(0,1).transpose(0,2).contiguous().unsqueeze(0)
print('patchdim',good_patch.shape)
print('img2dim',img2.shape)
'''
print('gpshape',good_patch.shape)
nps = calc_nps(good_patch, printability_array)
print('nps',nps)
tv = total_variation(img)
print(tv)
tv2 = total_variation(img2)
print(tv2)
'''
truths = read_truths('inria/Train/labels/crop001002.txt')
#false_truth = [0.0,0.2,0.3,0.4,0.5]
#truths = np.vstack((truths,false_truth))
truths = torch.from_numpy(truths)
print(truths)
print('good_patch.type()',good_patch.type())
apply_patch(img2,good_patch,truths)

