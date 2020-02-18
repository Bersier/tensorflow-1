from src.commons.imports import tf

ones = tf.ones(shape=(2, 3, 4))
twos = 2 * ones

mask = tf.constant(((True, False, False), (False, True, False)), dtype=tf.dtypes.bool)
print(mask)
mask = tf.broadcast_to(tf.expand_dims(mask, axis=2), shape=(2, 3, 4))
print(mask)

mixed = tf.where(mask, ones, twos)

print(mixed)
