import os 

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import io
import tensorflow as tf 
import matplotlib.pyplot as plt
import numpy as np

from tensorflow import keras 


# Make sure we don't get any GPU errors
physical_devices = tf.config.list_physical_devices("GPU")
print(len(physical_devices))
#tf.config.experimental.set_memory_growth(physical_devices[0], True)

