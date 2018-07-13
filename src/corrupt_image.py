from scipy.misc import imread, imsave, imresize
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', type=str, required=True)
parser.add_argument('-o', '--output', type=str, required=True)
parser.add_argument('-t', '--type', type=str, 
                    choices=(['gaussian_noise',
                              'quantize',
                              'sr_linear',
                              'sr_nn',
                              'sr_black'
                             ]),
                    default='gaussian_noise')

def tofloat(img):
    img = img.astype(np.float32)
    return img/np.max(img)

def gaussian_noise(img, std=0.1):
    i = tofloat(img)
    i = i + std*np.random.normal(size=img.shape)
    return i

def quantize_2(img, levels=4):
    i = tofloat(img)
    i = np.floor(i*levels)/levels
    return i

def quantize(img, quantize_factor=55):
    img = img.astype(float)
    img /= 255.0
    img *= (255.0 / quantize_factor)
    images = img
    images = np.floor(images)
    images /= (255.0 / quantize_factor)
    images *= 255.0
    images = images.astype(int)
    return images


def resize(img, rate=4., interp='bilinear'):
    return imresize(img, float(rate), interp=interp) 

def sr_linear(img, rate=4.):
    s = resize(img, 1./rate, interp='bilinear')
    return resize(s, rate, interp='bilinear')

def sr_nn(img, rate=4.):
    s = resize(img, 1./rate, interp='bilinear')
    return resize(s, rate, interp='nearest')

def sr_black(img, rate=4):
    o = np.zeros_like(img)
    s = resize(img, 1./rate, interp='bilinear')
    o[::rate, ::rate, :] = s
    return o

def main(args):
    i = imread(args.input)
    o = eval(args.type)(i)
    imsave(args.output, o)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
    
