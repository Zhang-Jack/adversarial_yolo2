import math
import torch
import numpy as np
from craft_adv import read_and_size_image
import torch.nn.functional as F
from torch.nn import ConstantPad2d
import torchvision.transforms.functional as tvfunc


def transform_patch(adv_patch, angle, scale):
    # angle converteren naar radialen
    print(scale)
    angle = math.pi / 180 * angle
    # maak een mask die ook getransformeerd kan worden (om de padding later te kunnen verwijderen)
    mask = torch.ones(adv_patch.size())
    # pad adv_patch zodat de gedraaide adv_patch niet buiten de grenzen van de afbeelding valt
    p_height = adv_patch.size(2)
    p_width = adv_patch.size(3)
    padding = (math.sqrt((p_height/2)**2+(p_width/2)**2)-max(p_height,p_width)/2)*abs(np.sin(2*angle))+(scale-1)/4*p_height
    print('padding',padding)
    mypad = ConstantPad2d(math.ceil(padding*scale), 0)
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
    print(rot_patch.shape)
    # zorg dat de padding naar waarde 2 gezet wordt 
    rot_patch[rot_mask==0] = 2
    return rot_patch
    
if __name__ == '__main__':
    adv_patch = read_and_size_image('data/horse.jpg')
    angle = 90
    rot_patch = transform_patch(adv_patch, angle, 0.5)
    rot_patch[rot_patch==2] = 0.5
    zien = tvfunc.to_pil_image(rot_patch.squeeze(0))
    zien.show()
    #zien2 = tvfunc.to_pil_image(adv_patch.squeeze(0))
    #zien2.show()
