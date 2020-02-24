from tensorflow.keras import initializers
from tensorflow.keras.layers import InputSpec

from src.commons.imports import tf
from src.commons.tensorflow.operation import batch_dot_product


class DoubleDense(tf.keras.layers.Layer):
    def __init__(self,
                 unit_count,
                 wee_kernel_initializer,
                 big_kernel_initializer,
                 bias_initializer=initializers.zeros):
        super(DoubleDense, self).__init__()

        self.unit_count = unit_count
        self.wee_kernel_initializer = wee_kernel_initializer
        self.big_kernel_initializer = big_kernel_initializer
        self.bias_initializer = bias_initializer

        self.supports_masking = False
        self.input_spec = InputSpec(ndim=3)

    # noinspection PyAttributeOutsideInit
    def build(self, input_shape):
        _, b, s = input_shape

        self.wee_kernel = self._add_weight("wee_kernel", (s, self.unit_count), self.wee_kernel_initializer)
        self.wee_bias = self._add_weight("wee_bias", (self.unit_count,), self.bias_initializer)
        self.big_kernel = self._add_weight("big_kernel", (b, self.unit_count), self.big_kernel_initializer)
        self.big_bias = self._add_weight("big_bias", (self.unit_count,), self.bias_initializer)

        self.built = True

    def _add_weight(self, name, shape, initializer):
        return self.add_weight(
            name=name,
            shape=shape,
            initializer=initializer,
            dtype=self.dtype,
            trainable=True
        )

    # noinspection PyMethodOverriding
    def call(self, inputs):
        x = tf.matmul(a=inputs, b=self.wee_kernel)
        x += tf.broadcast_to(self.wee_bias, shape=(1,) + x.shape[1:])

        x = batch_dot_product(
            x,
            tf.expand_dims(self.big_kernel, axis=0),
            axis=1
        )
        x += self.big_bias

        return x

    def compute_output_shape(self, input_shape):
        return input_shape[0].concatenate(self.unit_count)

    def get_config(self):
        config = {
            'unit_count': self.unit_count,
            'wee_kernel_initializer': initializers.serialize(self.wee_kernel_initializer),
            'big_kernel_initializer': initializers.serialize(self.big_kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
        }
        base_config = super(DoubleDense, self).get_config()
        return dict(base_config.items() + config.items())
