import sys
import time
import os
import math
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

    # transform the output tensor from [batch, 425, 19, 19] to [batch, 80, 1805]
    print(output.shape)
    output = output.view(batch, 5, 5+num_classes, h*w)   #[batch, 5, 85, 361]
    print(output.shape)
    output = output.transpose(1,2).contiguous() #[batch, 85, 5, 361]
    print(output.shape)
    output = output.view(batch, 5+num_classes, 5*h*w)    #[batch, 85, 1805]
    print(output.shape)
    output_confs = output[:, 5:5+num_classes, :]    #[batch, 80, 1805]
    output_objectness = torch.sigmoid(output[:, 4, :])
    print('objectness', output_objectness.shape)
    print(output_confs.shape)

    # perform softmax to normalize probabilities for object classes to [0,1]
    normal_confs = torch.nn.Softmax(dim=1)(output_confs)
    print(normal_confs.shape)
    # we only care for probabilities of the class of interest (person)
    confs_for_class = normal_confs[:,cls_id,:]
    print(confs_for_class.shape)
    confs_if_object = confs_for_class*output_objectness
    # find the max probability for person
    max_conf, max_conf_idx = torch.max(confs_if_object, dim=1)
    print(max_conf.shape)
    print(max_conf)
    return max_conf

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
    return torch.from_numpy(printability_array).double()    #veranderen naar float?

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
    # bepaal het aantal pixels in adv_patch
    numel = torch.numel(adv_patch)/3            # delen door 3 want 3 kleurenkanalen
    
    # bereken de total variation van de adv_patch
    tv  = torch.sum(torch.abs(adv_patch[:, :, 1:] - adv_patch[:, :, :-1])) 
    tv += torch.sum(torch.abs(adv_patch[:, 1:, :] - adv_patch[:, :-1, :]))
    
    # normaliseer de total variation naar het aantal pixels in de patch
    # grotere patches worden dan niet extra afgestraft
    tv = tv/numel
    return tv

def read_and_size_image(imgfile, width=None, height=None):
    '''
    Leest het beeld met path imgfile in als PIL Image en transformeert het 
    naar een tensor met dimensie [B x C x H x W] met waarden in [0,1]
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

def transform_patch(adv_patch, scale):
    '''
    Transformeert de adversarial patch in adv_patch (Tensor) door deze te draaien 
    over een random hoek en te scalen met factor scale
    adv_patch:  tensor met dim [channels][adv_height][adv_width]
    scale:      int met de schaal om te resizen
    '''
    # contrast en brightness transformeren
    contrast = round(random.uniform(0.7, 1.0), 10)
    brightness = round(random.uniform(-0.5, 0.2), 10)
    adv_patch = adv_patch*contrast+brightness
    # ruis toevoegen
    noise = torch.rand(adv_patch.size())-0.5
    adv_patch += random.uniform(0.0, 0.3)*noise
    # zorg dat de waarden van het beeld binnen de range blijven
    adv_patch[adv_patch>1]=1
    adv_patch[adv_patch<0]=0
    # angle converteren naar radialen
    angle = random.uniform(0.0, 2*math.pi)
    # maak een mask die ook getransformeerd kan worden (om de padding later te kunnen verwijderen)
    mask = torch.ones(adv_patch.size())
    # pad adv_patch zodat de gedraaide adv_patch niet buiten de grenzen van de afbeelding valt
    p_height = adv_patch.size(2)
    p_width = adv_patch.size(3)
    ##print('p_height',p_height,'p_width',p_width)
    padding = (math.sqrt((p_height/2)**2+(p_width/2)**2)-max(p_height,p_width)/2)*abs(np.sin(2*angle))+(scale-1)/4*p_height
    if padding<0: padding=0    
    ##print('padding',padding)
    mypad = nn.ConstantPad2d(math.ceil(padding*scale), 0)
    padded_patch = mypad(adv_patch)
    padded_mask = mypad(mask)
    # construeer een affine_grid dat de transformatie zal uitvoeren
    theta = torch.zeros(1, 2, 3)
    theta[:, :, :2] = torch.FloatTensor([[np.cos(angle), np.sin(angle)],
                                         [-np.sin(angle),  np.cos(angle)]])
    theta[:, :, 2] = 0
    theta = theta/scale
    grid = F.affine_grid(theta,padded_patch.size())
    # voer de rotatie uit door grid_sample te doen
    rot_patch = F.grid_sample(padded_patch, grid, padding_mode='zeros')
    rot_mask = F.grid_sample(padded_mask, grid, padding_mode='zeros')
    ##print(rot_patch.shape)
    # zorg dat de padding naar waarde 2 gezet wordt 
    rot_patch[rot_mask==0] = 2
    return rot_patch

def apply_patch(img, adv_patch, truths):
    '''
    Plaatst de adversarial patch in adv_patch (Tensor) op het beeld img (Tensor) 
    op basis van de locatie van de bounding boxes in truths. 
    Voegt een willekeurige rotatie toe aan de patch.
    adv_patch:  tensor met dim [channels][adv_height][adv_width]
    img:        tensor met dim [channels][img_height][img_width]
    truths:     array van bbox-locaties zoals gelezen uit .txt-file
    '''
    #verwijder thruths die niet van class 0 zijn
    truths = truths[truths[:,0]==0]

    #haal de kolom met classes weg
    truths = truths[:,1:]

    #converteer thruths naar pixellocaties op basis van de dimensies van img
    truths[:,0] *= img.size(3)
    truths[:,1] *= img.size(2)
    truths[:,2] *= img.size(3)
    truths[:,3] *= img.size(2)

    #transformeer adv_patches en plaats ze op de afbeelding
    patched_img = img
    for truth in truths:        # batchen
        #bereken de dimensie van adv_patch a.d.h.v. de thruth
        patch_dim = int(math.sqrt((truth[2].mul(0.2).int())**2+(truth[3].mul(0.2).int())**2))

        #transformeer de patch
        scale = patch_dim/adv_patch.size(2)
        ##print('patch_dim', patch_dim)
        ##print('adv_patch.size(2)',adv_patch.size(2))
        ##print('scale', scale)
        transformed_patch = transform_patch(adv_patch, scale)

        #bepaal de padding van de patch (om van de patch een beeld te maken dat even groot is als het inputbeeld)
        int_truth = truth.int()
        lpad = int(int_truth[0]-(transformed_patch.size(2)/2))
        rpad = img.size(3)-lpad-transformed_patch.size(2)
        tpad = int(int_truth[1]-(transformed_patch.size(2)/2))
        bpad = img.size(2)-tpad-transformed_patch.size(2)

        #voer deze padding uit
        mypad = nn.ConstantPad2d((lpad,rpad,tpad,bpad), 2)
        padded_patch = mypad(transformed_patch)

        #voeg de patch toe aan het inputbeeld
        patched_img[padded_patch!=2] = padded_patch[padded_patch!=2]
    return patched_img

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

    imgfile2 = 'inria/Train/pos/crop001002.png'
    sized = read_and_size_image(imgfile,darknet_model.width,darknet_model.height)
    sized2 = read_and_size_image(imgfile2, darknet_model.width, darknet_model.height)
    img2 = read_and_size_image(imgfile2)
    sized3 = torch.randn(sized2.shape)

    batch = torch.cat([sized, sized2, sized3], dim=0)
    #sized = img.resize((darknet_model.width, darknet_model.height))

    # move the darknet model to the GPU
    darknet_model = darknet_model
    sized = torch.autograd.Variable(batch)
    output = darknet_model.forward(batch)
    get_max_probability(output, 0, 80)

'''deze wegdoen
img_interp = F.interpolate(img, size=(200,200), mode='bilinear', align_corners=True)

#zien = tvfunc.to_pil_image(img_interp.squeeze(0))
#zien.show()

printability_file = 'non_printability/30values.txt'
img_height = 500
img_width = 500
printability_array = get_printability_array(printability_file, img_height, img_width)
#good_patch = torch.from_numpy(np.tile([0.7098,0.32157,0.2],(img_height,img_width,1))).float()
#good_patch = good_patch.view(img_height, img_width, 3).transpose(0,1).transpose(0,2).contiguous().unsqueeze(0)
good_patch = read_and_size_image('data/horse.jpg')
#deze wegdoen'''
'''
print('gpshape',good_patch.shape)
nps = calc_nps(good_patch, printability_array)
print('nps',nps)
tv = total_variation(img)
print(tv)
tv2 = total_variation(img2)
print(tv2)
'''
'''deze wegdoen
truths = read_truths('inria/Train/pos/yolo-labels/crop001002.txt')
#false_truth = [0.0,0.2,0.3,0.4,0.5]
#truths = np.vstack((truths,false_truth))
truths = torch.from_numpy(truths)
patched_img = apply_patch(img2,good_patch,truths)
#laat de gepatchte afbeelding zien
zien = tvfunc.to_pil_image(patched_img.squeeze(0))
zien.show()
deze wegdoen'''

