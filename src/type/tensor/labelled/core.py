import tensorflow as tf

# See also http://nlp.seas.harvard.edu/NamedTensor (just found this link).
from src.commons.python.core import to_list, index_map
from src.type.tensor.labelled.axis import Dynamic


class LabelledTensor:

    @classmethod
    def new(cls, tensor, labels):
        indices = index_map(labels)
        return LabelledTensor(tensor, labels, indices)

    def __init__(self, tensor, labels, indices):  # TODO Where to store lengths dictionary?
        self._tensor = tensor
        self._labels = labels
        self._indices = indices

    def raw(self):
        return self._tensor

    def __add__(self, other):
        return LabelledTensor(self._tensor + self.__align(other), self._labels, self._indices)

    def __mul__(self, other):
        return LabelledTensor(self._tensor * self.__align(other), self._labels, self._indices)

    def concat(self, other, self_axis, other_axis, new_label=None):
        if not new_label:
            new_label = Dynamic()
        aligned_other = self.__align_along_axis(other, self_axis, other_axis)
        self_index = self._indices[self_axis]
        new_labels = self._labels.copy()
        new_labels[self_index] = new_label
        return LabelledTensor.new(tf.concat([self._tensor, aligned_other], axis=self_index), new_labels)

    def split(self, axis, index, new_label_1, new_label_2):
        axis_index = self._indices[axis]
        length_1 = index
        length_2 = self._tensor.shape[axis_index] - length_1
        tensor_1, tensor_2 = tf.split(self._tensor, [length_1, length_2], axis=axis_index)
        new_labels_1 = self._labels.copy()
        new_labels_2 = self._labels.copy()
        new_labels_1[axis_index] = new_label_1
        new_labels_2[axis_index] = new_label_2
        return LabelledTensor.new(tensor_1, new_labels_1), LabelledTensor.new(tensor_2, new_labels_2)

    def set_shape(self, shape):  # TODO ensure_shape in test mode
        self._tensor.set_shape(to_list(shape, self._indices, len(self._labels)))

    def check_shape(self, shape):  # TODO ensure_shape in test mode
        for label, length in shape:
            axis_length = self._tensor.shape[self._indices[label]]
            if not axis_length == length:
                raise ValueError("Shape check failed. Length along", label, "is", axis_length, "rather than", length)

    def __align(self, other):
        # noinspection PyProtectedMember
        return self.__align_helper(other, other._labels, other._indices)

    def __align_along_axis(self, other, self_axis, other_axis):
        # noinspection PyProtectedMember
        other_indices = other._indices.copy()
        # noinspection PyProtectedMember
        other_labels = other._labels.copy()
        axis_index = other_indices.pop(other_axis)
        other_indices[self_axis] = axis_index
        other_labels[axis_index] = self_axis
        return self.__align_helper(other, other_labels, other_indices)

    def __align_helper(self, other, other_labels, other_indices):
        if other_labels == self._labels:
            # noinspection PyProtectedMember
            return other._tensor
        elif other_indices.keys() == self._indices.keys():
            permutation = [other_indices[label] for label in self._labels]
            # noinspection PyProtectedMember
            return tf.transpose(a=other._tensor, perm=permutation)
        else:
            # noinspection PyProtectedMember
            raise ValueError("Axis mismatch. Self labels:", self._labels, "Other labels:", other._labels)
