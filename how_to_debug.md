Before comparing the middle results, please use png images instead of jpeg images without resize.

#### Pytorch-yolo2
1. set model to evaluation
```
m = m.eval()
```
2. In darknet.py, uncomment if condition to stop forward on certain layer
```
    def forward(self, x):
        ind = -2
        self.loss = None
        outputs = dict()
        for block in self.blocks:
            ind = ind + 1
            #if ind > 27:
            #    break
```
3. In utils.py:do_detect, add
```
print(output.storage()[0:100])
```
to get the layer output

4. Prepare a test image, resized to 416x416, save to test.png. Stop resize
in detect.py:detect
```
sized = img.resize((m.width, m.height)) ->
sized = img
```

#### Darknet
1. set GPU and CUDNN to 0 in Makefile

2. In src/detector.c:test_dector, stop resize
```
image sized = letterbox_image(im, net.w, net.h);  ->
image sized = im
```
And add output print
```
layer ll = net.layers[27];
float *Y = ll.output;
printf("---- Y ----\n");
for(j = 0; j < 169; j++) printf("%d: %f\n", j, Y[j]);
printf("\n");
```

#### Reorg Problem
There seems the problem in darknet

#### get_region_boxes speed up
detect.py cfg/yolo.cfg yolo.weight data/dog.jpg
- slow : 0.145544 
- fast : 0.050640
- faster: 0.009280

train.py
- slow: 380ms
- fast: 114ms
- faster: 22ms (batch=64 1.5s)
- fasterer: gpu to cpu  (batch=64 0.15s)


### Region_Loss Debug
```
from __future__ import print_function
import torch.optim as optim
import os
import torch
import numpy as np
from darknet import Darknet
from PIL import Image
from utils import image2torch, convert2cpu
from torch.autograd import Variable

cfgfile = 'face4.1re_95.91.cfg'
weightfile = 'face4.1re_95.91.weights'
imgpath = 'data/train/images/10002.png'
labpath = imgpath.replace('images', 'labels').replace('JPEGImages', 'labels').replace('.jpg', '.txt').replace('.png','.txt')
label = torch.zeros(50*5)
if os.path.getsize(labpath):
    tmp = torch.from_numpy(np.loadtxt(labpath))
    #tmp = torch.from_numpy(read_truths_args(labpath, 8.0/img.width))
    #tmp = torch.from_numpy(read_truths(labpath))
    tmp = tmp.view(-1)
    tsz = tmp.numel()
    #print('labpath = %s , tsz = %d' % (labpath, tsz))
    if tsz > 50*5:
        label = tmp[0:50*5]
    elif tsz > 0:
        label[0:tsz] = tmp
label = label.view(1, 50*5)

m = Darknet(cfgfile)
region_loss = m.loss
m.load_weights(weightfile)
m.eval()
m = m.cuda()

optimizer = optim.SGD(m.parameters(), lr=1e-4, momentum=0.9)

img = Image.open(imgpath)
img = image2torch(img)
img = Variable(img.cuda())

target = Variable(label)

print('----- img ---------------------')
print(img.data.storage()[0:100])
print('----- target  -----------------')
print(target.data.storage()[0:100])

optimizer.zero_grad()
output = m(img)
print('----- output ------------------')
print(output.data.storage()[0:100])
print(output.data.storage()[1000:1100])
loss = region_loss(output, target)
print('----- loss --------------------')
print(loss)

save_grad = None
def extract(grad):
    global saved_grad
    saved_grad = convert2cpu(grad.data)

output.register_hook(extract)
loss.backward()

saved_grad = saved_grad.view(-1)
for i in xrange(saved_grad.size(0)):
    if abs(saved_grad[i]) >= 0.001:
        print('%d : %f' % (i, saved_grad[i]))
```
### SGD debug
```
import torch
from torch.autograd import Variable
x = torch.rand(1,1)
x = Variable(x)
m = torch.nn.Linear(1,1)
lr = 0.1 
momentum = 0.9 
decay = 0 # 0.0005
optimizer = torch.optim.SGD(m.parameters(), lr=0.1, momentum=0.9, weight_decay=decay)
optimizer.zero_grad()

y = m(x)

w = m.weight.data.clone()
gw = (2*x*y).data.clone()
print('x = %f, y = %f' % (x.data[0][0], y.data[0][0]))
print('before: m.weight = %f, m.bias = %f' % (m.weight.data[0][0], m.bias.data[0]))
loss = y**2
loss.backward()
optimizer.step()
print('after: m.weight = %f, m.bias = %f' % (m.weight.data[0][0], m.bias.data[0]))
print('m.weight.grad = %f, m.bias.grad = %f' % (m.weight.grad.data[0][0], m.bias.grad.data[0]))
print('x = %f, y = %f' % (x.data[0][0], y.data[0][0]))

state_w = 0 
state_w = state_w * momentum + gw + decay*w
ww = w - lr*state_w
print('w: grad %f  %f -> %f' % (gw[0][0], w[0][0], ww[0][0]))
```
