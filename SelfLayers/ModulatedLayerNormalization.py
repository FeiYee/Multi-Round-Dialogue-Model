from keras.layers import *
from keras_layer_normalization import LayerNormalization

from ReuseLayer import ReuseLayer

class SelfModulatedLayerNormalization(ReuseLayer):
    """
    模仿Self-Modulated Batch Normalization，
    将Batch Normalization改为Layer Normalization
    """

    def __init__(self, num_hidden, **kwargs):
        super(SelfModulatedLayerNormalization, self).__init__(**kwargs)
        self.num_hidden = num_hidden

    def build(self, input_shape):
        super(SelfModulatedLayerNormalization, self).build(input_shape)
        output_dim = input_shape[0][-1]
        self.layernorm = LayerNormalization(center=False, scale=False)
        self.beta_dense_1 = Dense(self.num_hidden, activation='relu')
        self.beta_dense_2 = Dense(output_dim)
        self.gamma_dense_1 = Dense(self.num_hidden, activation='relu')
        self.gamma_dense_2 = Dense(output_dim)

    def call(self, inputs):
        inputs, cond = inputs
        inputs = self.reuse(self.layernorm, inputs)
        beta = self.reuse(self.beta_dense_1, cond)
        beta = self.reuse(self.beta_dense_2, beta)
        gamma = self.reuse(self.gamma_dense_1, cond)
        gamma = self.reuse(self.gamma_dense_2, gamma)
        for _ in range(K.ndim(inputs) - K.ndim(cond)):
            beta = K.expand_dims(beta, 1)
            gamma = K.expand_dims(gamma, 1)
        return inputs * (gamma + 1) + beta

    def compute_output_shape(self, input_shape):
        return input_shape[0]