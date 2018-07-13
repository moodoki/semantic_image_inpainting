from scipy.misc import imread
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', type=str, help="original")
parser.add_argument('-o', '--output', type=str, help="processed")

def psnr(a, b):
    a = normalize(a)
    b = normalize(b)
    e = a-b
    n = np.mean(np.multiply(e, e))
    return 10*np.log(1/n)/np.log(10)

def normalize(i):
    i = i.astype(np.float32)
    return i/np.max(i)

def main(args):
    a = imread(args.input)
    b = imread(args.output)
    print(args.output, psnr(a, b))

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

    
