import abc

import tensorflow as tf


# class Data(abc.ABC):
#     @abc.abstractmethod
#     def price_history(self):
#         pass
#
#
# class NormalizedData(abc.ABC):
#     def __init__(self, log_returns):
#         self._log_returns = log_returns
#         self.mean, self.std = tf.nn.moments(log_returns, axes=1)
#
#     def log_returns(self):
#         return self._log_returns
#
#     def mean(self) -> tf.Tensor:
#         return self.mean
#
#     def std(self) -> tf.Tensor:
#         return self.std
from src.types.axes import Axis


class Context(abc.ABC):
    @abc.abstractmethod
    def length(self, axe: Axis) -> int:
        pass


class ExampleIterator:
    def __init__(self, dataset, c):
        iterator = tf.compat.v1.data.make_initializable_iterator(dataset)

        self._batch_count = c.length(Axis.Batch)
        self._initializer = iterator.initializer
        self._next_inputs_seq, self._next_shared_input_seq = iterator.get_next()

        self._next_inputs_seq.set_shape((None, c.length(Axis.Time), c.length(Axis.Stock), c.length(Axis.Feature)))
        self._next_shared_input_seq.set_shape((None, c.length(Axis.Time), c.length(Axis.SharedFeature)))
        # TODO use ensure_shape for testing

    def initializer(self):
        return self._initializer

    def next_inputs_seq(self):
        return self._next_inputs_seq

    def next_shared_input_seq(self):
        return self._next_shared_input_seq

    def batch_count(self):
        return self._batch_count


class __AbsData(abc.ABC):
    @abc.abstractmethod
    def iterator(self) -> ExampleIterator:
        pass


class WindowedData(__AbsData, abc.ABC):
    pass


class Data(__AbsData, abc.ABC):

    @abc.abstractmethod
    def windowed(self, window_length) -> WindowedData:
        pass


class PriceHistoryData(Data, WindowedData):
    def __init__(self, price_history: tf.data.Dataset):
        # TODO make log returns
        self._dataset = price_history
        pass

    def iterator(self) -> ExampleIterator:
        return ExampleIterator(self._dataset, None)

    def windowed(self, window_length) -> 'PriceHistoryData':
        return PriceHistoryData(self._dataset.flat_map(
            lambda *seq:
            tf.data.Dataset.from_tensor_slices(seq)
                .window(window_length, shift=1, drop_remainder=True)
                .flat_map(lambda inputs, shared_input: tf.data.Dataset.zip((inputs, shared_input))
                          .batch(window_length))
        ))


class SimpleQuandlData(PriceHistoryData):
    pass


class NextExample:
    pass
