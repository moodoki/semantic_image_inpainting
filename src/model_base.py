"""Model base class."""

import tensorflow as tf
import numpy as np
import tools
import abc

class ModelBase(object):
  def __init__(self, modelfilename, config,
               model_name='dcgan',
               gen_input='z:0', gen_output='Tanh:0', gen_loss='Mean_2:0',
               disc_input='real_images:0', disc_output='Sigmoid:0',
               z_dim=100, batch_size=64):
    """
    Model for Semantic image inpainting.
    Loads frozen weights of a GAN and create the graph according to the
    loss function as described in paper

    Args:
      modelfilename: tensorflow .pb file with weights to be loaded
      config: training parameters: lambda_p, nIter
      gen_input: node name for generator input
      gen_output: node name for generator output
      disc_input: node name for discriminator input
      disc_output: node name for discriminator output
      z_dim: latent space dimension of GAN
      batch_size: training batch size
    """
    __metaclass__ = abc.ABCMeta
    self.config = config

    self.batch_size = batch_size
    self.z_dim = z_dim
    self.graph, self.graph_def = tools.loadpb(modelfilename,
    model_name)

    self.gi = self.graph.get_tensor_by_name(model_name+'/'+gen_input)
    self.go = self.graph.get_tensor_by_name(model_name+'/'+gen_output)
    self.gl = self.graph.get_tensor_by_name(model_name+'/'+gen_loss)
    self.di = self.graph.get_tensor_by_name(model_name+'/'+disc_input)
    self.do = self.graph.get_tensor_by_name(model_name+'/'+disc_output)

    self.image_shape = self.go.shape[1:].as_list()

    self.l = config.lambda_p

    self.sess = tf.Session(graph=self.graph)

    self.init_z()

  def init_z(self):
    """Initializes latent variable z"""
    self.z = np.random.randn(self.batch_size, self.z_dim)
    #self.z = np.random.uniform(size=[self.batch_size, self.z_dim])

  def sample(self, z=None):
    """GAN sampler. Useful for checking if the GAN was loaded correctly"""
    if z is None:
      z = self.z
    sample_out = self.sess.run(self.go, feed_dict={self.gi: z})
    return sample_out

  @abc.abstractmethod
  def preprocess(self, **kwargs):
    """
    """
    pass

  @abc.abstractmethod
  def postprocess(self, g_out, **kwargs):
    """
    """
    pass

  @abc.abstractmethod
  def build_context_loss(self):
    """Builds context loss function.
    """
    pass

  @abc.abstractmethod
  def build_input_placeholders(self):
    pass

  @abc.abstractmethod
  def restore_image(self, image, **kwargs):
    """
    """
    pass

  def build_restore_graph(self):
    """
    """
    self.build_input_placeholders()
    self.build_context_loss()

    self.perceptual_loss = self.gl
    self.inpaint_loss = self.context_loss + self.l*self.perceptual_loss
    self.inpaint_grad = tf.gradients(self.inpaint_loss, self.gi)


  def backprop_to_input(self, verbose=True):
    """Main worker function. To be called after all initilization is done.

    Performs backpropagation to input using (accelerated) gradient descent
    to obtain latent space representation of target image

    Returns:
      generator output image
    """
    v = 0
    for i in range(self.config.nIter):
      out_vars = [self.inpaint_loss, self.inpaint_grad, self.go]
      if hasattr(self, 'masks_data'):
        in_dict = {self.masks: self.masks_data,
                   self.gi: self.z,
                   self.images: self.images_data}
      else:
        in_dict = {self.gi: self.z,
                   self.images: self.images_data}


      loss, grad, imout = self.sess.run(out_vars, feed_dict=in_dict)

      v_prev = np.copy(v)
      v = self.config.momentum*v - self.config.lr*grad[0]
      self.z += (-self.config.momentum * v_prev +
                 (1 + self.config.momentum) * v)
      self.z = np.clip(self.z, -1, 1)

      if verbose:
        print('Iteration {}: {}'.format(i, np.mean(loss)))
    return imout
