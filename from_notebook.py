import PIL

print(PIL.PILLOW_VERSION)
import load_data
from load_data import *
import load_data
import gc
import matplotlib.pyplot as plt
from torch import autograd
import patch_config

plt.rcParams["axes.grid"] = False
plt.axis('off')

img_dir = "inria/Train/pos"
lab_dir = "inria/Train/pos/yolo-labels"
cfgfile = "cfg/yolov2.cfg"
weightfile = "weights/yolov2.weights"
printfile = "non_printability/30values.txt"
patch_size = 300
mode = "exp1"
config = patch_config.patch_configs[mode]()

print('LOADING MODELS')
darknet_model = Darknet(cfgfile)
darknet_model.load_weights(weightfile)
darknet_model = darknet_model.eval().cuda()
patch_applier = PatchApplier().cuda()
patch_transformer = PatchTransformer().cuda()
prob_extractor = MaxProbExtractor(0, 80, config).cuda()
nps_calculator = NPSCalculator(printfile, patch_size)
nps_calculator = nps_calculator.cuda()
total_variation = TotalVariation().cuda()
print('MODELS LOADED')

img_size = darknet_model.height
batch_size = 6  # 10#18
n_epochs = 10000
max_lab = 14

# Choose between initializing with gray or random
adv_patch_cpu = torch.full((3, patch_size, patch_size), 0.5)
# adv_patch_cpu = torch.rand((3,patch_size,patch_size))


patch_img = Image.open("saved_patches/patchnew0.jpg").convert('RGB')
tf = transforms.Resize((patch_size, patch_size))
patch_img = tf(patch_img)
tf = transforms.ToTensor()
adv_patch_cpu = tf(patch_img)

adv_patch_cpu.requires_grad_(True)

print('INITIALIZING DATALOADER')
train_loader = torch.utils.data.DataLoader(InriaDataset(img_dir, lab_dir, max_lab, img_size, shuffle=True),
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=10)
print('DATALOADER INITIALIZED')

optimizer = optim.Adam([adv_patch_cpu], lr=.03, amsgrad=True)

# try:
et0 = time.time()
for epoch in range(n_epochs):
    ep_det_loss = 0
    bt0 = time.time()
    for i_batch, (img_batch, lab_batch) in enumerate(train_loader):
        with autograd.detect_anomaly():
            img_batch = img_batch.cuda()
            lab_batch = lab_batch.cuda()
            # print('TRAINING EPOCH %i, BATCH %i'%(epoch, i_batch))
            adv_patch = adv_patch_cpu.cuda()
            adv_batch_t = patch_transformer(adv_patch, lab_batch, img_size, do_rotate=True)
            p_img_batch = patch_applier(img_batch, adv_batch_t)
            p_img_batch = F.interpolate(p_img_batch, (darknet_model.height, darknet_model.width))
            output = darknet_model(p_img_batch)
            max_prob = prob_extractor(output)
            nps = nps_calculator(adv_patch)
            tv = total_variation(adv_patch)

            det_loss = torch.mean(max_prob)
            ep_det_loss += det_loss.detach().cpu().numpy()
            '''
            nps_loss = nps
            tv_loss = tv*8
            loss = nps_loss + (det_loss**3/tv_loss + tv_loss**3/det_loss)**(1/3)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            adv_patch_cpu.data.clamp_(0,1)       #keep patch in image range

            '''
            nps_loss = nps * 0.01
            tv_loss = tv * 2.5
            loss = det_loss + nps_loss + tv_loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            adv_patch_cpu.data.clamp_(0, 1)  # keep patch in image range

            bt1 = time.time()
            if i_batch % 5 == 0:
                print('BATCH', i_batch, end='...\n')
                im = transforms.ToPILImage('RGB')(adv_patch_cpu)
                plt.imshow(im)
                plt.show()
            '''
            print('  BATCH NR: ', i_batch)
            print('BATCH LOSS: ', loss.detach().cpu().numpy())
            print('  DET LOSS: ', det_loss.detach().cpu().numpy())
            print('  NPS LOSS: ', nps_loss.detach().cpu().numpy())
            print('   TV LOSS: ', tv_loss.detach().cpu().numpy())
            print('BATCH TIME: ', bt1-bt0)
            '''
            if i_batch + 1 >= len(train_loader):
                print('\n')
            else:
                del adv_batch_t, output, max_prob, det_loss, p_img_batch, nps_loss, tv_loss, loss
                torch.cuda.empty_cache()
            bt0 = time.time()
    et1 = time.time()
    ep_det_loss = ep_det_loss / len(train_loader)
    ep_nps_loss = nps_loss.detach().cpu().numpy()
    ep_tv_loss = tv_loss.detach().cpu().numpy()
    tot_ep_loss = ep_det_loss + ep_nps_loss + ep_tv_loss
    if True:
        print('  EPOCH NR: ', epoch),
        print('EPOCH LOSS: ', tot_ep_loss)
        print('  DET LOSS: ', ep_det_loss)
        print('  NPS LOSS: ', ep_nps_loss)
        print('   TV LOSS: ', ep_tv_loss)
        print('EPOCH TIME: ', et1 - et0)
        im = transforms.ToPILImage('RGB')(adv_patch_cpu)
        plt.imshow(im)
        plt.show()
        im.save("saved_patches/patchnew1.jpg")
        del adv_batch_t, output, max_prob, det_loss, p_img_batch, nps_loss, tv_loss, loss
        torch.cuda.empty_cache()
    et0 = time.time()