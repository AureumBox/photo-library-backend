# %%
# Imports
import os
import numpy as np
from keras import layers
from tensorflow import data as tf_data
import matplotlib.pyplot as plt
import tensorflow as tf


# %%
# Filter out corrupted images
num_skipped = 0
for folder_name in (
    "animals",
    "architecture",
    "battles",
    "bookcovers",
    "book_pages",
    "foods",
    "landscapes",
    "maps",
    "paintings",
    "people",
    "plants",
    "rivers",
    "sculptures",
    "stamps",
):
    folder_path = os.path.join("training_dataset", folder_name)
    for fname in os.listdir(folder_path):
        fpath = os.path.join(folder_path, fname)
        try:
            fobj = open(fpath, "rb")
            is_jfif = b"JFIF" in fobj.peek(10)
        finally:
            fobj.close()

        if not is_jfif:
            num_skipped += 1
            # Delete corrupted image
            os.remove(fpath)

print(f"Deleted {num_skipped} images.")

# %%
# Generate a Dataset
image_size = (180, 180)
batch_size = 20

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "training_dataset",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)

# %%
# Using image data augmentation
data_augmentation_layers = [
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
]


def data_augmentation(images):
    for layer in data_augmentation_layers:
        images = layer(images)
    return images


plt.figure(figsize=(10, 10))
for images, _ in train_ds.take(1):
    for i in range(9):
        augmented_images = data_augmentation(images)
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(np.array(augmented_images[0]).astype("uint8"))
        plt.axis("off")

# %%
# Visualize the data
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(np.array(images[i]).astype("uint8"))
        plt.title(int(labels[i]))
        plt.axis("off")

# %%
# Configure the dataset for performance
# Apply `data_augmentation` to the training images.
train_ds = train_ds.map(
    lambda img, label: (data_augmentation(img), label),
    num_parallel_calls=tf_data.AUTOTUNE,
)
# Prefetching samples in GPU memory helps maximize GPU utilization.
train_ds = train_ds.prefetch(tf_data.AUTOTUNE)


# %%
# Modelo MobileNet
num_classes = 14
class_names = [
    "animals",
    "architecture",
    "battles",
    "bookcovers",
    "book_pages",
    "foods",
    "landscapes",
    "maps",
    "paintings",
    "people",
    "plants",
    "rivers",
    "sculptures",
    "stamps",
]

mobilenet_model = tf.keras.applications.MobileNetV2(
    weights="imagenet", input_shape=(image_size[0], image_size[1], 3), include_top=False
)
mobilenet_model.trainable = False  # freezing

inputs = tf.keras.Input(shape=(image_size[0], image_size[1], 3))
x = mobilenet_model(inputs, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
outputs = tf.keras.layers.Dense(num_classes)(x)

model = tf.keras.Model(inputs, outputs)

# %%
# Training
model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)
model.fit(train_ds, epochs=10)

# guardar
model.save("models/classifier1.h5")

"""
LUEGO PARA USARLO
from tensorflow import keras

# Cargar el modelo
model = keras.models.load_model('models/classifier1.h5')
"""

# %%
# Probar lmao
img = tf.keras.preprocessing.image.load_img(
    "C:\\Users\\aurim\Desktop\sitios-fantasticos-descubrir-bolivar-fc12c3d903a6a32f6a61daf37f8fa7cf.jpg",
    target_size=image_size,
)
plt.imshow(img)

img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # Create batch axis

predictions = model.predict(img_array)
predicted_class = np.argmax(predictions[0])  # Get the index of the highest score

# Define your class names according to your categories
print(
    f"This image is most likely {class_names[predicted_class]} with a {100 * np.max(predictions[0]):.2f}% confidence."
)

# %%
