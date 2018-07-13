import tensorflow as tf
import numpy as np
import external.poissonblending as blending
from scipy.signal import convolve2d
from PIL import Image


class ModelColorize():
    def __init__(self, modelfilename, config,
                 model_name='dcgan',
                 gen_input='z:0', gen_output='Tanh:0', gen_loss='Mean_2:0',
                 disc_input='real_images:0', disc_output='Sigmoid:0',
                 z_dim=100, batch_size=64):
        """
        Model for Semantic image inpainting.
        Loads frozen weights of a GAN and create the graph according to the
        loss function as described in paper

        Arguments:
            modelfilename - tensorflow .pb file with weights to be loaded
            config - training parameters: lambda_p, nIter
            gen_input - node name for generator input
            gen_output - node name for generator output
            disc_input - node name for discriminator input
            disc_output - node name for discriminator output
            z_dim - latent space dimension of GAN
            batch_size - training batch size
        """

        self.config = config

        self.batch_size = batch_size
        self.z_dim = z_dim
        self.graph, self.graph_def = ModelColorize.loadpb(modelfilename,
                                                         model_name)

        self.gi = self.graph.get_tensor_by_name(model_name+'/'+gen_input)
        self.go = self.graph.get_tensor_by_name(model_name+'/'+gen_output)
        self.gl = self.graph.get_tensor_by_name(model_name+'/'+gen_loss)
        self.di = self.graph.get_tensor_by_name(model_name+'/'+disc_input)
        self.do = self.graph.get_tensor_by_name(model_name+'/'+disc_output)

        self.image_shape = self.go.shape[1:-1].as_list() + [1]

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

    def preprocess(self, images):
        """Default preprocessing pipeline
        Converts input image from single channel to 3 channel grayscale
        Prepare the data to be fed to the network. Weighted mask is computed
        and images and masks are duplicated to fill the batch.

        Arguments:
            image - input image

        Returns:
            None
        """
        images = ModelColorize.imtransform(images)

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

    def postprocess(self, g_out, blend=False):
        """Default post processing pipeline
        Currently does nothing

        Arguments:
            g_out - generator output
        """
        images_out = ModelColorize.iminvtransform(g_out)
        images_in = ModelColorize.iminvtransform(self.images_data)

        if blend:
            for idx, (i, o) in enumerate(zip(images_in, images_out)):
                images_out[idx, :, :, :] = ModelColorize.colorblend( i, o )

        return images_out

    def build_colorization_graph(self):
        """Builds the context and prior loss objective"""
        with self.graph.as_default():
            self.images = tf.placeholder(tf.float32,
                                         [None] + self.image_shape,
                                         name='images')
            self.context_loss = tf.reduce_sum(
                    tf.contrib.layers.flatten(
                        tf.abs(tf.image.rgb_to_grayscale(self.go) -
                               self.images)), 1
                )

            self.prior_loss = self.gl
            self.colorization_loss = self.context_loss + self.l*self.prior_loss
            self.colorization_grad = tf.gradients(self.colorization_loss, self.gi)

    def colorize(self, image, blend=False):
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
        self.build_colorization_graph()
        self.preprocess(image)

        imout = self.backprop_to_input()

        return self.postprocess(imout, blend), imout

    def backprop_to_input(self, verbose=True):
        """Main worker function. To be called after all initilization is done.
        Performs backpropagation to input using (accelerated) gradient descent
        to obtain latent space representation of target image

        Returns:
            generator output image
        """
        v = 0
        for i in range(self.config.nIter):
            out_vars = [self.colorization_loss, self.colorization_grad, self.go]
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

    @staticmethod
    def loadpb(filename, model_name='dcgan'):
        """Loads pretrained graph from ProtoBuf file

        Arguments:
            filename - path to ProtoBuf graph definition
            model_name - prefix to assign to loaded graph node names

        Returns:
            graph, graph_def - as per Tensorflow definitions
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

    @staticmethod
    def imtransform(img):
        """Helper: Rescale pixel value ranges to -1 and 1"""
        return np.array(img) / 127.5-1

    @staticmethod
    def iminvtransform(img):
        """Helper: Rescale pixel value ranges to 0 and 1"""
        return (np.array(img) + 1.0) / 2.0

    @staticmethod
    def poissonblending(img1, img2, mask):
        """Helper: interface to external poisson blending"""
        return blending.blend(img1, img2, 1 - mask)

    @staticmethod
    def colorblend(img_gray, img_color):
        """Helper to apply color from one image to another"""
        img_blended_hsv = np.zeros_like(img_color, dtype=np.uint8)
        img_blended_hsv[:,:,2:] = np.uint8(np.copy(img_gray*255))
        img_color_hsv = np.array(Image.fromarray(np.uint8(img_color*255), mode='RGB').convert('HSV'))
        img_blended_hsv[:,:,:2] = img_color_hsv[:,:,:2]
        img_out = np.array(Image.fromarray(img_blended_hsv, mode='HSV').convert('RGB'))

        return img_out
