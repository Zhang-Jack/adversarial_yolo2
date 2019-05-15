import math
import torch
import numpy as np
import torch
from craft_adv import read_and_size_image
import torchvision.transforms.functional as tvfunc

def th_iterproduct(*args):
    return torch.from_numpy(np.indices(args).reshape((len(args),-1)).T)

def th_bilinear_interp2d(input, coords):
    """
    bilinear interpolation in 2d
    """
    print('input.shape',input.shape)
    print('coords.shape',coords.shape)
    x = torch.clamp(coords[:,:,0], 0, input.size(1)-2)
    x0 = x.floor()
    x1 = x0 + 1
    y = torch.clamp(coords[:,:,1], 0, input.size(2)-2)
    y0 = y.floor()
    y1 = y0 + 1

    print('x.shape',x.shape)
    stride = torch.tensor(input.stride(), dtype=x.dtype)
    x0_ix = x0.mul(stride[1].float()).long()
    x1_ix = x1.mul(stride[1].float()).long()
    y0_ix = y0.mul(stride[2].float()).long()
    y1_ix = y1.mul(stride[2].float()).long()
    
    
    input_flat = input.view(input.size(0),-1)

    vals_00 = input_flat.gather(1, x0_ix.add(y0_ix))
    vals_10 = input_flat.gather(1, x1_ix.add(y0_ix))
    vals_01 = input_flat.gather(1, x0_ix.add(y1_ix))
    vals_11 = input_flat.gather(1, x1_ix.add(y1_ix))
    print('vals_00.shape',vals_00.shape)    

    xd = x - x0
    yd = y - y0
    xm = 1 - xd
    ym = 1 - yd

    x_mapped = (vals_00.mul(xm).mul(ym) +
                vals_10.mul(xd).mul(ym) +
                vals_01.mul(xm).mul(yd) +
                vals_11.mul(xd).mul(yd))
    print('x_mapped.shape',x_mapped.shape)
    print('input.shape',input.shape)
    return x_mapped.view_as(input)

def getRotationMatrix(theta):
    theta = math.pi / 180 * theta
    print('theta',theta)
    
    rotation_matrix = torch.FloatTensor([[math.cos(theta), -math.sin(theta), 0],
                                         [math.sin(theta), math.cos(theta), 0],
                                         [0, 0, 1]])

    return rotation_matrix

def th_affine2d(x, matrix, mode='bilinear', center=True):
    """
    2D Affine image transform on th.Tensor
    
    Arguments
    ---------
    x : th.Tensor of size (C, H, W)
        image tensor to be transformed
    matrix : th.Tensor of size (3, 3) or (2, 3)
        transformation matrix
    mode : string in {'nearest', 'bilinear'}
        interpolation scheme to use
    center : boolean
        whether to alter the bias of the transform 
        so the transform is applied about the center
        of the image rather than the origin
    Example
    ------- 
    >>> import torch
    >>> from torchsample.utils import *
    >>> x = th.zeros(2,1000,1000)
    >>> x[:,100:1500,100:500] = 10
    >>> matrix = th.FloatTensor([[1.,0,-50],
    ...                             [0,1.,-50]])
    >>> xn = th_affine2d(x, matrix, mode='nearest')
    >>> xb = th_affine2d(x, matrix, mode='bilinear')
    """
    print('matrix',matrix)
    if matrix.dim() == 2:
        matrix = matrix[:2,:]
        matrix = matrix.unsqueeze(0)
    elif matrix.dim() == 3:
        if matrix.size()[1:] == (3,3):
            matrix = matrix[:,:2,:]
    print('matrix.shape',matrix.shape)
    
    A_batch = matrix[:,:,:2]
    print('A_batch',A_batch)
    if A_batch.size(0) != x.size(0):
        A_batch = A_batch.repeat(x.size(0),1,1)
    b_batch = matrix[:,:,2].unsqueeze(1)
    print('b_batch',b_batch)
    
    # make a meshgrid of normal coordinates
    _coords = th_iterproduct(x.size(1),x.size(2))
    coords = _coords.unsqueeze(0).repeat(x.size(0),1,1).float()

    if center:
        # shift the coordinates so center is the origin
        coords[:,:,0] = coords[:,:,0] - (x.size(1) // 2. - 0.5)
        coords[:,:,1] = coords[:,:,1] - (x.size(2) // 2. - 0.5)
    # apply the coordinate transformation
    new_coords = coords.bmm(A_batch.transpose(1,2)) + b_batch.expand_as(coords)

    if center:
        # shift the coordinates back so origin is origin
        new_coords[:,:,0] = new_coords[:,:,0] + (x.size(1) // 2. - 0.5)
        new_coords[:,:,1] = new_coords[:,:,1] + (x.size(2) // 2. - 0.5)

    # map new coordinates using bilinear interpolation
    if mode == 'nearest':
        x_transformed = th_nearest_interp2d(x.contiguous(), new_coords)
    elif mode == 'bilinear':
        x_transformed = th_bilinear_interp2d(x.contiguous(), new_coords)

    return x_transformed

if __name__ == '__main__':
    imgfile = 'inria/Train/pos/crop001002.png'
    img = read_and_size_image(imgfile)
    print('img.shape',img.shape)
    img = img.squeeze()
    print('img.shape',img.shape)
    mat = getRotationMatrix(20)
    rotated_img = th_affine2d(img, mat)
    zien = tvfunc.to_pil_image(rotated_img)
    zien.show()
    
