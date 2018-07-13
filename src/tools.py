"""Helper functions
"""
import tensorflow as tf
import numpy as np
import external.poissonblending as blending
from scipy.signal import convolve2d

###################Helpers for image inapinting #######################
def imtransform(img):
	"""Helper: Rescale pixel value ranges to -1 and 1"""
	return np.array(img) / 127.5-1

def iminvtransform(img):
	"""Helper: Rescale pixel value ranges to 0 and 1"""
	return (np.array(img) + 1.0) / 2.0

def poissonblending(img1, img2, mask):
	"""Helper: interface to external poisson blending"""
	return blending.blend(img1, img2, 1 - mask)

def createWeightedMask(mask, nsize=7):
	"""Takes binary weighted mask to create weighted mask as described in paper.
	Args:
		mask: binary mask input. numpy float32 array
		nsize: pixel neighbourhood size. default = 7
	"""
	ker = np.ones((nsize,nsize), dtype=np.float32)
	ker = ker/np.sum(ker)
	wmask = mask * convolve2d(1-mask, ker, mode='same', boundary='symm')
	return wmask

def binarizeMask(mask, dtype=np.float32):
	"""Helper function, ensures mask is 0/1 or 0/255 and single channel.

	If dtype specified as float32 (default), output mask will be 0, 1
	if required dtype is uint8, output mask will be 0, 255

	Args:
		mask:.
		dtype:.
	"""
	assert(np.dtype(dtype) == np.float32 or np.dtype(dtype) == np.uint8)
	bmask = np.array(mask, dtype=np.float32)
	bmask[bmask>0] = 1.0
	bmask[bmask<=0] = 0
	if dtype == np.uint8:
		bmask = np.array(bmask*255, dtype=np.uint8)
	return bmask

def create3ChannelMask(mask):
	"""Helper function, repeats single channel mask to 3 channels"""
	assert(len(mask.shape)==2)
	return np.repeat(mask[:,:,np.newaxis], 3, axis=2)

def loadpb(filename, model_name='dcgan'):
  """Loads pretrained graph from ProtoBuf file
  Args:
    filename: path to ProtoBuf graph definition.
    model_name: prefix to assign to loaded graph node names.
  Returns:
    graph, graph_def: as per Tensorflow definitions.
  """
  with tf.gfile.GFile(filename, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

  with tf.Graph().as_default() as graph:
    tf.import_graph_def(graph_def,
                        input_map=None,
                        return_elements=None,
                        op_dict=None,
                        producer_op_list=None,
                        name=model_name)

  return graph, graph_def
