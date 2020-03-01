from keras.layers import *
from keras import backend as K

from ReuseLayer import ReuseLayer

class Bidirectional(ReuseLayer):
    """
    双向RNN，允许传入mask，保证对齐
    """

    def __init__(self, layer, **args):
        super(Bidirectional, self).__init__(**args)
        self.forward_layer = copy.deepcopy(layer)
        self.backward_layer = copy.deepcopy(layer)
        self.forward_layer.name = 'forward_' + self.forward_layer.name
        self.backward_layer.name = 'backward_' + self.backward_layer.name

    def reverse_sequence(self, x, mask):
        """这里的mask.shape是[batch_size, seq_len, 1]
        """
        seq_len = K.round(K.sum(mask, 1)[:, 0])
        seq_len = K.cast(seq_len, 'int32')
        return K.tf.reverse_sequence(x, seq_len, seq_dim=1)

    def call(self, inputs):
        x, mask = inputs
        x_forward = self.reuse(self.forward_layer, x)
        x_backward = self.reverse_sequence(x, mask)
        x_backward = self.reuse(self.backward_layer, x_backward)
        x_backward = self.reverse_sequence(x_backward, mask)
        x = K.concatenate([x_forward, x_backward], 2)
        return x * mask

    def compute_output_shape(self, input_shape):
        return (None, input_shape[0][1], self.forward_layer.units * 2)