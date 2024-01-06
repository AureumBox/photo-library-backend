# Esto debe cambiar mucho pero shh

# %%
import os
import numpy as np
import keras
from keras import layers
from tensorflow import data as tf_data
import matplotlib.pyplot as plt

# %%
# Filter out corrupted images
num_skipped = 0
for folder_name in (
    "Animals",
    "Architecture",
    "Battles",
    "Bookcovers",
    "BookPages",
    "Foods",
    "Landscapes",
    "Maps",
    "Paintings",
    "People",
    "Plants",
    "Rivers",
    "Sculptures",
    "Stamps",
):
    folder_path = os.path.join("DatasetDictionary", folder_name)
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

train_ds = keras.preprocessing.image_dataset_from_directory(
    "DatasetDictionary",
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
import tensorflow as tf

mobilenet_model = tf.keras.applications.MobileNetV2(
    weights="imagenet", input_shape=(image_size[0], image_size[1], 3), include_top=False
)


mobilenet_model.trainable = False  # freezing


num_classes = 14

inputs = keras.Input(shape=(150, 150, 3))

x = mobilenet_model(inputs, training=False)

x = keras.layers.GlobalAveragePooling2D()(x)

outputs = keras.layers.Dense(num_classes)(x)
model = keras.Model(inputs, outputs)


# %%
# Training
model.compile(    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(train_ds, epochs=10)

# guardar
model.save('models/classifier1.h5')

'''
LUEGO PARA USARLO
from tensorflow import keras

# Cargar el modelo
model = keras.models.load_model('models/classifier1.h5')
'''

# %%
# Probar lmao
img = keras.preprocessing.image.load_img("C:\Users\aurim\Desktop\sitios-fantasticos-descubrir-bolivar-fc12c3d903a6a32f6a61daf37f8fa7cf.jpg", target_size=image_size)
plt.imshow(img)

img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # Create batch axis

predictions = model.predict(img_array)
predicted_class = np.argmax(predictions[0])  # Get the index of the highest score

# Define your class names according to your categories
class_names = ["Animals", "Architecture", "Battles", "Bookcovers", "BookPages", "Foods", "Landscapes", "Maps", "Paintings", "People", "Plants", "Rivers", "Sculptures", "Stamps"]

print(f"This image is most likely {class_names[predicted_class]} with a {100 * np.max(predictions[0]):.2f}% confidence.")


"""
    # %%
    # Build a model
    def make_model(input_shape, num_classes):
        inputs = keras.Input(shape=input_shape)

        # Entry block
        x = layers.Rescaling(1.0 / 255)(inputs)
        x = layers.Conv2D(128, 3, strides=2, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        previous_block_activation = x  # Set aside residual

        for size in [256, 512, 728]:
            x = layers.Activation("relu")(x)
            x = layers.SeparableConv2D(size, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)

            x = layers.Activation("relu")(x)
            x = layers.SeparableConv2D(size, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)

            x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

            # Project residual
            residual = layers.Conv2D(size, 1, strides=2, padding="same")(
                previous_block_activation
            )
            x = layers.add([x, residual])  # Add back residual
            previous_block_activation = x  # Set aside next residual

        x = layers.SeparableConv2D(1024, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        x = layers.GlobalAveragePooling2D()(x)
        if num_classes == 2:
            units = 1
        else:
            units = num_classes

        x = layers.Dropout(0.25)(x)
        # We specify activation=None so as to return logits
        outputs = layers.Dense(units, activation=None)(x)
        return keras.Model(inputs, outputs)


    model = make_model(input_shape=image_size + (3,), num_classes=2)
    keras.utils.plot_model(model, show_shapes=True)

    # %%
    # Train the model
    epochs = 25

    callbacks = [
        keras.callbacks.ModelCheckpoint("save_at_{epoch}.keras"),
    ]
    model.compile(
        optimizer=keras.optimizers.Adam(3e-4),
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[keras.metrics.BinaryAccuracy(name="acc")],
    )
    model.fit(
        train_ds,
        epochs=epochs,
        callbacks=callbacks,
        validation_data=val_ds,
    )

    # %%
    # Run inference on new data
    img = keras.utils.load_img("PetImages/Cat/6779.jpg", target_size=image_size)
    plt.imshow(img)

    img_array = keras.utils.img_to_array(img)
    img_array = keras.ops.expand_dims(img_array, 0)  # Create batch axis

    predictions = model.predict(img_array)
    score = float(keras.ops.sigmoid(predictions[0][0]))
    print(f"This image is {100 * (1 - score):.2f}% cat and {100 * score:.2f}% dog.")

    #%%

"""
