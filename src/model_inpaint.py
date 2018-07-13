import tensorflow as tf
import numpy as np
import tools

from model_base import ModelBase

class ModelInpaint(ModelBase):
    def preprocess(self, images, imask, useWeightedMask=True, nsize=15):
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
        if useWeightedMask:
            mask = tools.createWeightedMask(imask, nsize)
        else:
            mask = imask
        mask = tools.create3ChannelMask(mask)

        bin_mask = tools.binarizeMask(imask, dtype='uint8')
        self.bin_mask = tools.create3ChannelMask(bin_mask)

        self.masks_data = np.repeat(mask[np.newaxis, :, :, :],
                                    self.batch_size,
                                    axis=0)

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

        if blend:
            for i in range(len(g_out)):
                images_out[i] = tools.poissonblending(
                    images_in[i], images_out[i], self.bin_mask
                )
        else:
            images_out = np.multiply(images_out, 1-self.masks_data) \
                         + np.multiply(images_in, self.masks_data)

        return images_out

    def build_input_placeholders(self):
      with self.graph.as_default():
        self.masks = tf.placeholder(tf.float32,
                                    [None] + self.image_shape,
                                    name='mask')
        self.images = tf.placeholder(tf.float32,
                                     [None] + self.image_shape,
                                     name='images')

    def build_context_loss(self):
        """Builds the context and prior loss objective"""
        with self.graph.as_default():
          self.context_loss = tf.reduce_sum(
            tf.contrib.layers.flatten(
              tf.abs(tf.multiply(self.masks, self.go) -
                     tf.multiply(self.masks, self.images))), 1
          )

    def restore_image(self, image, mask, blend=True):
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
