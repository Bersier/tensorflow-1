from tensorflow.keras import initializers
from tensorflow.keras.layers import InputSpec

from src.commons.imports import tf


class OptionCase(tf.keras.layers.Layer):
    """
    Implements a layer that is the equivalent of a case statement for missing features (Options).

    Assumptions:
      - The input is nested.
      - The last axis iterates over features of the nested vectors.
      - All nested vectors are of the same type (stuctural, and distributional).
      - A nested vector can be present (Some), or absent (None).
      - The presence of a vector is represented by absence of nans:
        - If none of the features of a nested vector are nan, then that vector is present.
        - If any of the the features of a nested vector are nan, then that vector is absent.

    """

    def __init__(self,
                 repr_length,
                 none_initializer=initializers.zeros,
                 kernel_initializer=initializers.lecun_normal(),
                 bias_initializer=initializers.zeros):
        super(OptionCase, self).__init__()

        self.repr_length = repr_length
        self.none_initializer = none_initializer
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer

        self.supports_masking = False
        self.input_spec = InputSpec(min_ndim=2)

    # noinspection PyAttributeOutsideInit
    def build(self, input_shape):
        s = input_shape[-1]

        self.none_repr = self._add_weight("none_repr", (self.repr_length,), self.none_initializer)
        self.kernel = self._add_weight("kernel", (s, self.repr_length), self.kernel_initializer)
        self.bias = self._add_weight("bias", (self.repr_length,), self.bias_initializer)

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
        last_input_axis = len(inputs.shape) - 1
        a_axes = (last_input_axis,)
        b_axes = (0,)

        x = tf.tensordot(a=inputs, b=self.kernel, axes=(a_axes, b_axes))

        x += tf.broadcast_to(self.bias, shape=(1,) + x.shape[1:])
        # x += tf.reshape(self.bias, shape=(1,) * last_input_axis + (self.repr_length,))

        nan_mask = tf.math.is_nan(inputs)
        nan_mask = tf.reduce_any(nan_mask, axis=last_input_axis)

        nan_mask = tf.expand_dims(nan_mask, axis=last_input_axis)
        nan_mask = tf.broadcast_to(nan_mask, x.shape)
        # nan_mask = broadcast_along(nan_mask, x.shape, axes=[last_input_axis])

        return tf.where(nan_mask, x, self.none_repr)

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.repr_length,)

    def get_config(self):
        config = {
            'unit_count': self.repr_length,
            'none_initializer': initializers.serialize(self.none_initializer),
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
        }
        base_config = super(OptionCase, self).get_config()
        return dict(base_config.items() + config.items())
