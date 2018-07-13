import tensorflow as tf
import scipy.misc
import argparse
import os
import numpy as np
from glob import glob
from helper import loadimage_gray, saveimages

from model_colorize import ModelColorize

parser = argparse.ArgumentParser()
parser.add_argument('--model_file', type=str, help="Pretrained GAN model")
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--nIter', type=int, default=1000)
parser.add_argument('--imgSize', type=int, default=64)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lambda_p', type=float, default=0.1)
parser.add_argument('--checkpointDir', type=str, default='checkpoint')
parser.add_argument('--outDir', type=str, default='colorization')
parser.add_argument('--blend', action='store_true', default=False,
                    help="Blend predicted image to original image")
parser.add_argument('--in_image', type=str, default=None,
                    help='Input Image (ignored if inDir is specified')
parser.add_argument('--inDir', type=str, default=None,
                    help='Path to input images')
parser.add_argument('--imgExt', type=str, default='png',
                    help='input images file extension')
parser.add_argument('-c', action='store_true', help='corrupt image on the fly')

args = parser.parse_args()

def main():
    m = ModelColorize(args.model_file, args)

    # Generate some samples from the model as a test
    imout = m.sample()
    #saveimages(imout)

    if args.inDir is not None:
        imgfilenames = glob( args.inDir + '/*.' + args.imgExt )
        print('{} images found'.format(len(imgfilenames)))
        in_img = np.array([loadimage_gray(f) for f in imgfilenames])
    elif args.in_img is not None:
        in_img = loadimage_gray(args.in_image)
    else:
        print('Input image needs to be specified')
        exit(1)

    saveimages(in_img, 'colorize_in', imgfilenames, args.outDir)

    inpaint_out, g_out = m.colorize(in_img, args.blend)
    saveimages(g_out, 'colorize_gen', imgfilenames, args.outDir)
    saveimages(inpaint_out, 'colorize', imgfilenames, args.outDir)


if __name__ == '__main__':
    main()
