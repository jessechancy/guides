import os
import jax
import tensorflow as tf
import torch
import tqdm
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras_core
import keras_nlp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import seaborn as sns

print(os.listdir("."))

print(jax.random.normal(jax.random.PRNGKey(0), (3000,3000), dtype=jax.numpy.float32))

print(tf.constant([1, 2, 3]))

print(torch.rand(3, 3).size())

for _ in tqdm.tqdm(range(5)): pass

print(keras_core.models.Sequential([keras_core.layers.Dense(2)]).summary())
print(keras_nlp.__version__)

plt.plot([1, 2, 3], [1, 4, 9])
plt.savefig("matplotlib.png")


print(np.random.rand(3, 3).shape)
print(pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]}).head())
Image.new('RGB', (50, 50), color = 'red').save('pillow.png')
sns.lineplot(x=[1,2,3], y=[1,4,9])
plt.savefig("seaborn.png")
