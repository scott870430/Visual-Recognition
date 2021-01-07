import argparse, os
import torch
from torch.autograd import Variable
from scipy.ndimage import imread
import scipy
from PIL import Image
import numpy as np
import time, math
import matplotlib.pyplot as plt
import cv2

parser = argparse.ArgumentParser(description="PyTorch VDSR Demo")
parser.add_argument("--cuda", action="store_true", help="use cuda?")
parser.add_argument("--model", default="model/model_epoch_50.pth", type=str, help="model path")
parser.add_argument("--image", default="butterfly_GT", type=str, help="image name")
parser.add_argument("--scale", default=3, type=int, help="scale factor, Default: 3")
parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")
parser.add_argument("--output", default="output", type=str, help="output file name")
parser.add_argument("--path", default="../dataset/testing_lr_images/", type=str, help="output file name")

def PSNR(pred, gt, shave_border=0):
    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)
    
def colorize(y, ycbcr): 
    img = np.zeros((y.shape[0], y.shape[1], 3), np.uint8)
    img[:,:,0] = y
    img[:,:,1] = ycbcr[:,:,1]
    img[:,:,2] = ycbcr[:,:,2]
    img = Image.fromarray(img, "YCbCr").convert("RGB")
    return img

opt = parser.parse_args()
cuda = opt.cuda

if not os.path.isdir('./' + opt.output):
    os.makedirs('./' + opt.output)

if cuda:
    print("=> use gpu id: '{}'".format(opt.gpus))
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
    if not torch.cuda.is_available():
            raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

model = torch.load(opt.model, map_location=lambda storage, loc: storage)["model"]
if cuda:
    model = model.cuda()

img_path = opt.path
img_list = os.listdir(opt.path)
for img in img_list:
    print(img)
    im_b_ycbcr = imread(img_path + img, mode="YCbCr")
    w, h = im_b_ycbcr.shape[:2]
    #im_b_ycbcr = scipy.misc.imresize(im_b_ycbcr, (w * opt.scale, h * opt.scale))

    im_b_y = im_b_ycbcr[:,:,0].astype(float)


    im_input = im_b_y/255.

    im_input = Variable(torch.from_numpy(im_input).float()).view(1, -1, im_input.shape[0], im_input.shape[1])

    if cuda:
        im_input = im_input.cuda()
    else:
        model = model.cpu()


    out = model(im_input).cpu()

    im_h_y = out.data[0].numpy().astype(np.float32)

    im_h_y = im_h_y * 255.
    im_h_y[im_h_y < 0] = 0
    im_h_y[im_h_y > 255.] = 255.


    im_h = colorize(im_h_y[0,:,:], im_b_ycbcr)
    im_h.save('./' + opt.output +'/'+ str(img))
    #cv2.imwrite('./' + opt.output + str(img), im_h)

