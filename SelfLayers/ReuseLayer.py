from keras.layers import *

class ReuseLayer(Layer):
    """
    增加reuse方法，允许在定义Layer时调用现成的层
    """

    def reuse(self, layer, *args, **kwargs):
        if not layer.built:
            if len(args) > 0:
                layer.build(K.int_shape(args[0]))
            else:
                layer.build(K.int_shape(kwargs['inputs']))
            self._trainable_weights.extend(layer._trainable_weights)
            self._non_trainable_weights.extend(layer._non_trainable_weights)
        return layer.call(*args, **kwargs)
