from src.commons.imports import tf
from src.commons.tensorflow.extention import with_noise
from src.commons.tensorflow.maker import NAN

ones = tf.ones(shape=(2, 3, 4))
twos = 2 * ones

z = tf.bitwise.bitwise_and(tf.cast(ones, dtype=tf.dtypes.int32), tf.cast(twos, dtype=tf.dtypes.int32))
print("z:", z)

print(ones[0, 1:2, 2:])

mask = tf.constant(((True, False, False), (False, True, False)), dtype=tf.dtypes.bool)
print(mask)
mask = tf.broadcast_to(tf.expand_dims(mask, axis=2), shape=(2, 3, 4))
print(mask)

mixed = tf.where(mask, ones, twos)

print(mixed)

noisy_ones = with_noise(noise=NAN, noise_proportion=0.2)(ones)
print(noisy_ones)
print(tf.reduce_min(noisy_ones, axis=2))
print(tf.reduce_max(noisy_ones, axis=2))
