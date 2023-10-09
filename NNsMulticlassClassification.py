import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import linear, relu, sigmoid
%matplotlib widget
import matplotlib.pyplot as plt
plt.style.use('./deeplearning.mplstyle')

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

from public_tests import *

from autils import *
from lab_utils_softmax import plt_softmax
np.set_printoptions(precision=2)

# UNQ_C1
# GRADED CELL: my_softmax

def my_softmax(z):
    """ Softmax converts a vector of values to a probability distribution.
    Args:
      z (ndarray (N,))  : input data, N features
    Returns:
      a (ndarray (N,))  : softmax of z
    """
    ### START CODE HERE ###
    a = [0] * len(z)
    z_sum = 0.
    for i in range(len(z)):
        z_sum += np.exp(z[i])
    for j in range(len(z)):
        a[j] = np.exp(z[j]) / z_sum
    ### END CODE HERE ###
    return a


# UNQ_C2
# GRADED CELL: Sequential model

tf.random.set_seed(1234)  # for consistent results
model = Sequential(
    [
        ### START CODE HERE ###
        tf.keras.Input(shape=(400,)),
        Dense(25, activation='relu', name="L1"),
        Dense(15, activation='relu', name="L2"),
        Dense(10, activation='linear', name="L3")

        ### END CODE HERE ###
    ], name="my_model"
)

