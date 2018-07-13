import tensorflow as tf
import numpy as np
import tools

from model_inpaint import ModelInpaint

def p_standard_gaussian(z):

    return tf.exp( -.5 * tf.reduce_sum( tf.square(z), axis=1 ) )

class ModelInpaintTest(ModelInpaint):
    """Test of small modificaiton to prior loss"""

    def build_restore_graph(self):
        super(ModelInpaintTest, self).build_restore_graph()

        p = p_standard_gaussian
        self.perceptual_loss = (self.gl + tf.log(tf.clip_by_value(
            1-tf.exp(tf.clip_by_value(self.gl, -5, 5)), 1e-4, 99999)
        ))*p(self.gi)

        self.inpaint_loss = self.context_loss + self.l*self.perceptual_loss
        self.inpaint_grad = tf.gradients(self.inpaint_loss, self.gi)

