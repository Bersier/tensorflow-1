from tensorflow.keras import initializers
from tensorflow.keras.layers import InputSpec

from src.commons.imports import tf
from src.commons.tensorflow.checker import has_nan
from src.commons.tensorflow.operation import tensor_dot
from src.type.core import Along


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
        - If any of the features of a nested vector are nan, then that vector is absent.

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
    def call(self, inputs):  # TODO test on non-random input, to see whether simpleoption beats vanilla
        last_axis = len(inputs.shape) - 1

        nan_mask = has_nan(inputs, axis=last_axis)
        nan_mask = tf.expand_dims(nan_mask, axis=last_axis)

        # Needed, because otherwise, the weights become NaN at some point (tf bug?)...
        inputs = tf.where(nan_mask, tf.zeros_like(inputs), inputs)

        # Apply kernel.
        x = tensor_dot(
            Along(inputs, [last_axis]),
            Along(self.kernel, [0])
        )

        x_shape = tf.shape(x)

        # Apply bias.
        x += tf.broadcast_to(self.bias, shape=x_shape)

        output_nan_mask = tf.broadcast_to(nan_mask, x_shape)
        broadcast_none_repr = tf.broadcast_to(self.none_repr, shape=x_shape)
        return tf.where(output_nan_mask, broadcast_none_repr, x)

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
