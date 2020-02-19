"""
Normal Tensors are flat; they don't have depth. So nested structures cannot be represented explicitly.
This lack of recursion leads to a lack of compositionality.
An operation cannot be naturally applied to nested sub-tensors.

This class is an attempt at fixing this, by keeping track of the structure separately,
while internally representing everything as flat tensors.
"""
