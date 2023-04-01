import abc


class Model_Weights(abc.ABC):
    @property
    @abc.abstractmethod
    def train_preprocess(self):
        pass

    @property
    @abc.abstractmethod
    def inference_preprocess(self):
        pass
