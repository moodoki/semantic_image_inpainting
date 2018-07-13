import tensorflow as tf
import numpy as np
import tools

from model_base import ModelBase

class ModelSuperres(ModelBase):
    def preprocess(self, images, mask=None):
        """Default preprocessing pipeline
        Prepare the data to be fed to the network. Weighted mask is computed
        and images and masks are duplicated to fill the batch.

        Arguments:
            image - input image
            mask - input mask

        Returns:
            None
        """
        images = tools.imtransform(images)
        #Generate multiple candidates for completion if single image is given
        if len(images.shape) is 3:
            self.images_data = np.repeat(images[np.newaxis, :, :, :],
                                         self.batch_size,
                                         axis=0)
        elif len(images.shape) is 4:
            #Ensure batch is filled
            num_images = images.shape[0]
            self.images_data = np.repeat(images[np.newaxis, 0, :, :, :],
                                         self.batch_size,
                                         axis=0)
            ncpy = min(num_images, self.batch_size)
            self.images_data[:ncpy, :, :, :] = images[:ncpy, :, :, :].copy()

    def postprocess(self, g_out, blend = True):
        """Default post processing pipeline
        Applies poisson blending using binary mask. (default)

        Arguments:
            g_out - generator output
            blend - Use poisson blending (True) or alpha blending (False)
        """
        images_out = tools.iminvtransform(g_out)
        images_in = tools.iminvtransform(self.images_data)
        return images_out

    def build_input_placeholders(self):
      with self.graph.as_default():
        self.masks = tf.placeholder(tf.float32,
                                    [None] + self.image_shape,
                                    name='mask')
        self.images = tf.placeholder(tf.float32,
                                     [None] + self.image_shape,
                                     name='images')

    def perform_corruption(self, images):
      down_sample_factor = self.down_sample_factor

      ret = tf.image.resize_bilinear(
                images,
                [tf.shape(images)[1] // down_sample_factor,
                 tf.shape(images)[2] // down_sample_factor,
                ],
                align_corners=True)

      ret = tf.image.resize_bilinear(ret,
                                     [tf.shape(images)[1],
                                      tf.shape(images)[2]
                                     ],
                                     align_corners=True)
      return ret

    def build_context_loss(self):
        """Builds the context and prior loss objective"""
        with self.graph.as_default():
          self.context_loss = tf.reduce_sum(
            tf.contrib.layers.flatten(
              tf.abs(self.perform_corruption(self.go) -
                     self.perform_corruption(self.images))), 1
          )

    def restore_image(self, image, mask=None, blend=True):
        """Perform inpainting with the given image and mask with the standard
        pipeline as described in paper. To skip steps or try other pre/post
        processing, the methods can be called seperately.

        Arguments:
            image - input 3 channel image
            mask - input binary mask, single channel. Nonzeros values are
                   treated as 1
            blend - Flag to apply Poisson blending on output, Default = True

        Returns:
            post processed image (merged/blneded), raw generator output
        """
        self.build_restore_graph()
        self.preprocess(image, mask)

        imout = self.backprop_to_input()

        return self.postprocess(imout, blend), imout
