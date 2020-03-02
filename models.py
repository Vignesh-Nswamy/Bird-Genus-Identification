import tensorflow as tf

from tensorflow.keras import Sequential

from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import AveragePooling1D
from tensorflow.keras.layers import Dense


class ConvolutionalRNN:
    def __init__(self, conv_stacks=5):
        self.conv_stacks = conv_stacks
        self.kernel_regularizer = tf.keras.regularizers.l2(0.001)
        self.he_normal = tf.keras.initializers.he_normal(seed=0)
        self.lecun_normal = tf.keras.initializers.lecun_normal(seed=0)
        self.model = Sequential()

    def add_conv_stack(self):
        self.model.add(Conv1D(filters=56,
                              kernel_size=5,
                              kernel_regularizer=self.kernel_regularizer,
                              kernel_initializer=self.he_normal))
        self.model.add(BatchNormalization())
        self.model.add(LeakyReLU())
        self.model.add(MaxPooling1D(2))

    def get_model(self):
        self.model.add(Input((862, 128)))
        for i in range(self.conv_stacks):
            self.add_conv_stack()
        self.model.add(LSTM(96, return_sequences=False))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(1024, activation='relu', kernel_initializer=self.he_normal))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(512, activation='relu', kernel_initializer=self.he_normal))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(182, activation='softmax', kernel_initializer=self.lecun_normal))
        return self.model

class SimpleRNN:
    def __init__(self):
        self.kernel_regularizer = tf.keras.regularizers.l2(0.001)
        self.he_normal = tf.keras.initializers.he_normal(seed=0)
        self.lecun_normal = tf.keras.initializers.lecun_normal(seed=0)
        self.model = Sequential()

    def get_model(self):
        self.model.add(LSTM(256, input_shape=(862, 128), return_sequences=False))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(512, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.he_normal, activation='relu'))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(256, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.he_normal, activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(182, kernel_initializer=self.lecun_normal, activation='softmax'))
        return self.model

if __name__ == '__main__':
    ConvolutionalRNN().get_model().summary()
    SimpleRNN().get_model().summary()
