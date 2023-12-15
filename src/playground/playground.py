import numpy as np
import tensorflow as tf

# # Create a numpy array with dim 160 x 143 with random values
# a = np.random.rand(160, 143)

# print(a.ndim)
# print(a.shape)

# # Reshape the array to 1 x 160 x 143 x 1
# a = a.reshape(1, 160, 143, 1)

# print(a.ndim)
# print(a.shape)
# print(a)

b = np.random.rand(144, 160, 3)

print(b.shape)
# # Reshape the array to 160 x 143 x 1 keeping the first dimension
# b = b.reshape(160, 143, 1)
# print(b)

# Keep only the first channel
b = b[:, :, 0:1]

print(b.shape)
# print(b)

c = b.reshape(1, 144, 160, 1)

print(c.shape)
# print(c)



print(c)
con = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(1, 144, 160, 1))
c = con(c)
avg_pool_2d = tf.keras.layers.AveragePooling2D(pool_size=(2, 2),
    strides=(1, 1), padding="valid")
y = avg_pool_2d(c)
print(y)
