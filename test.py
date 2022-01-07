import argparse
import os
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import PIL.Image as pil_image

from models import FSRCNN
from utils import convert_ycbcr_to_rgb, preprocess, calc_psnr, calc_ssim

# funtion to process image
def process_image(image,model,name,scale_factor):
    image_width = (image.width // scale_factor) * scale_factor
    image_height = (image.height // scale_factor) * scale_factor

    hr = image.resize((image_width, image_height), resample=pil_image.BICUBIC)
    lr = hr.resize((hr.width // scale_factor, hr.height // scale_factor), resample=pil_image.BICUBIC)
    bicubic = lr.resize((lr.width * scale_factor, lr.height * scale_factor), resample=pil_image.BICUBIC)
    bicubic.save(name.replace('.', '_bicubic_x{}.'.format(scale_factor)))

    lr, _ = preprocess(lr, device)
    hr, _ = preprocess(hr, device)
    _, ycbcr = preprocess(bicubic, device)

    with torch.no_grad():
        preds = model(lr).clamp(0.0, 1.0)

    psnr = calc_psnr(hr, preds)
    ssim_score = calc_ssim(hr, preds)
    print('PSNR: {:.2f}'.format(psnr))
    print("SSIM: {}".format(ssim_score))
    preds = preds.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)

    output = np.array([preds, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])
    output = np.clip(convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)
    output = pil_image.fromarray(output)
    output.save(name.replace('.', '_fsrcnn_x{}.'.format(scale_factor)))

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-file', type=str, required=True)
    parser.add_argument('--image-file', type=str, required=False)
    # add arguments for the image directory
    parser.add_argument('--image-dir', type=str, required=False)
    parser.add_argument('--scale', type=int, default=3)
    args = parser.parse_args()

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = FSRCNN(scale_factor=args.scale).to(device)

    state_dict = model.state_dict()
    for n, p in torch.load(args.weights_file, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)

    model.eval()

    # if argument for the image directory is given
    if args.image_dir:
        # get names  of images in the directory and append the given directory with the image name
        image_names = [os.path.join(args.image_dir, x) for x in os.listdir(args.image_dir)]
        # get all the images in the directory
        images = [pil_image.open(args.image_dir + '/' + f).convert('RGB') for f in os.listdir(args.image_dir)]
        # iterate over all the images
        for name,image in zip(image_names,images):
           process_image(image,model,name,args.scale)
    elif args.image_file:
        image = pil_image.open(args.image_file).convert('RGB')
        process_image(image,model,args.image_file,args.scale)
    else:
        print('No image file or image directory given')

    
