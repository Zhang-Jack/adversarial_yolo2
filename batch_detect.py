import sys
import time
import os
from PIL import Image, ImageDraw
from utils import *
from darknet import Darknet

if __name__ == '__main__':
    if len(sys.argv) == 4:
        cfgfile = sys.argv[1]
        weightfile = sys.argv[2]
        imgdir = sys.argv[3]

    use_cuda = False
    darknet_model = Darknet(cfgfile)
    darknet_model.load_weights(weightfile)
    if use_cuda:
        darknet_model = darknet_model.cuda()

    # read in the label names associated with the darknet model    
    if darknet_model.num_classes == 20:
        namesfile = 'data/voc.names'
    elif darknet_model.num_classes == 80:
        namesfile = 'data/coco.names'
    else:
        namesfile = 'data/names'
    for imgfile in os.listdir(imgdir):
        if imgfile.endswith('.jpg') or imgfile.endswith('.png'):
            name = os.path.splitext(imgfile)[0]    #image name w/o extension
            txtname = name + '.txt'
            txtpath = os.path.abspath(os.path.join(imgdir, 'yolo-labels/', txtname))
            # open beeld en resize
            imgfile = os.path.abspath(os.path.join(imgdir, imgfile))
            img = Image.open(imgfile).convert('RGB')
            img = img.resize((darknet_model.width, darknet_model.height))
            boxes = do_detect(darknet_model, img, 0.5, 0.4, use_cuda)
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
            textfile.close()
                    
                    
                    
                    
