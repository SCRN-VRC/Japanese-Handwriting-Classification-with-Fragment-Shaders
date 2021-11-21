from tensorflow.keras import layers
from tensorflow import keras
from keras.preprocessing.image import load_img
from keras import backend as K

import matplotlib.pyplot as plt
import tensorflow_addons as tfa
import tensorflow as tf
import numpy as np
import pickle
import struct
import os

DATA_DIR = './data/'
MODEL_DIR = './model/'

learning_rate = 0.00075
weight_decay = 0.00005
batch_size = 64
num_epochs = 30

# generated stuff
gen_out = np.load(DATA_DIR + 'gen_out.npy')
fb = open(DATA_DIR + 'gen_labels.pkl','rb')
gen_labels = pickle.load(fb)
fb.close()

# handwritten stuff
gen_out2 = np.load(DATA_DIR + 'gen_out2.npy')
fb = open(DATA_DIR + 'gen_labels2.pkl','rb')
gen_labels2 = pickle.load(fb)
fb.close()

gen_out2 = np.concatenate((gen_out2, gen_out))
gen_labels2 = list(gen_labels2) + gen_labels

# Convert the string to ints
num2Char = {}
char2Num = {}
with open(DATA_DIR + 'jp_seq2text.tsv', encoding='utf-8') as f:
    for line in f:
        char, i = line.strip('\n').split('\t')
        char2Num[char] = i
        num2Char[i] = char
    f.close()

# Labels start at 3
gen_labels2 = np.asarray([int(char2Num.get(lb)) - 3 for lb in gen_labels2])
gen_labels2 = np.expand_dims(gen_labels2, axis=1)

# shuffle data
def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

gen_out2, gen_labels2 = unison_shuffled_copies(gen_out2, gen_labels2)

# test data split
char_num = len(num2Char)
val_indices = char_num

x_test, y_test = gen_out2[:val_indices], gen_labels2[:val_indices]
x_train, y_train = gen_out2[val_indices:], gen_labels2[val_indices:]

# validation data split
val_indices = char_num * 4

new_x_train, new_y_train = x_train[val_indices:], y_train[val_indices:]
x_val, y_val = x_train[:val_indices], y_train[:val_indices]

print(f"Training data samples: {len(new_x_train)}")
print(f"Validation data samples: {len(x_val)}")
print(f"Test data samples: {len(x_test)}")

image_size = 64
auto = tf.data.AUTOTUNE

data_augmentation = keras.Sequential(
    [layers.RandomCrop(image_size, image_size), layers.RandomFlip("horizontal"),],
    name="data_augmentation",
)

def make_datasets(images, labels, is_train=False):
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    if is_train:
        dataset = dataset.shuffle(batch_size * 10)
    dataset = dataset.batch(batch_size)
    # if is_train:
    #     dataset = dataset.map(
    #         lambda x, y: (data_augmentation(x), y), num_parallel_calls=auto
    #     )
    return dataset.prefetch(auto)


train_dataset = make_datasets(new_x_train, new_y_train, is_train=True)
val_dataset = make_datasets(x_val, y_val)
test_dataset = make_datasets(x_test, y_test)

def activation_block(x):
    x = layers.Activation("gelu")(x)
    x = layers.BatchNormalization()(x)
    return layers.Dropout(0.1)(x)


def conv_stem(x, filters: int, patch_size: int):
    x = layers.Conv2D(filters, kernel_size=patch_size, strides=patch_size)(x)
    return activation_block(x)


def conv_mixer_block(x, filters: int, kernel_size: int):
    # Depthwise convolution.
    x0 = x
    x = layers.DepthwiseConv2D(kernel_size=kernel_size, padding="same")(x)
    x = layers.Add()([activation_block(x), x0])  # Residual.

    # Pointwise convolution.
    x = layers.Conv2D(filters, kernel_size=1)(x)
    x = activation_block(x)

    return x


def get_conv_mixer_256_8(
    image_size=32, filters=256, depth=8, kernel_size=5, patch_size=2, num_classes=10
):
    """ConvMixer-256/8: https://openreview.net/pdf?id=TVHS5Y4dNvM.
    The hyperparameter values are taken from the paper.
    """
    inputs = keras.Input((image_size, image_size, 1))
    x = layers.Rescaling(scale=1.0 / 255)(inputs)

    # Extract patch embeddings.
    x = conv_stem(x, filters, patch_size)

    # ConvMixer blocks.
    for _ in range(depth):
        x = conv_mixer_block(x, filters, kernel_size)

    # Classification block.
    x = layers.GlobalAvgPool2D()(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return keras.Model(inputs, outputs)


def run_experiment(model):
    optimizer = tfa.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    )

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    checkpoint_filepath = "./tmp/checkpoint"
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
    )

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=num_epochs,
        callbacks=[checkpoint_callback, early_stopping],
    )

    #model.load_weights(checkpoint_filepath)
    
    _, accuracy = model.evaluate(test_dataset)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")

    return history, model

def predict(model, img, debug=False):
    img = np.asarray(img)[:,:,0]
    img = np.expand_dims(img, axis=0)
    probs = model.predict(img)
    top5 = np.argpartition(probs[0], -5)[-5:]
    top5 = np.flip(top5)
    if (debug):
        print('Top 5 predictions: ', end = '')
        for item in top5:
            print(num2Char.get(str(item + 3)), end=', ')
        print()
    return num2Char.get(str(top5[0] + 3))
        

def write_weights(array, dest, mode='ab'):
    with open(dest, mode) as f:
        if (len(array.shape) == 4):
            for i in range(0, len(array)):
                for j in range(0, len(array[0])):
                    for k in range(0, len(array[0][0])):
                        for l in range(0, len(array[0][0][0])):
                            f.write(struct.pack('f', array[i][j][k][l]))
        elif (len(array.shape) == 3):
            for i in range(0, len(array)):
                for j in range(0, len(array[0])):
                    for k in range(0, len(array[0][0])):
                        f.write(struct.pack('f', array[i][j][k]))
        elif (len(array.shape) == 2):
            for i in range(0, len(array)):
                for j in range(0, len(array[0])):
                        f.write(struct.pack('f', array[i][j]))
        elif (len(array.shape) == 1):
            for i in range(0, len(array)):
                f.write(struct.pack('f', array[i]))
        f.close()

conv_mixer_model = get_conv_mixer_256_8(image_size=image_size,
                                        filters=144,
                                        depth=4,
                                        kernel_size=5,
                                        num_classes=char_num)
    
if 0:
    history, conv_mixer_model = run_experiment(conv_mixer_model)
    conv_mixer_model.save_weights(MODEL_DIR + 'conv_mixer_100_4_0.h5')
else:
    conv_mixer_model.load_weights(MODEL_DIR + 'conv_mixer_144_4_13.h5')
    img = load_img('./input/input1.jpg')
    #predict(conv_mixer_model, img, debug=True)
    
    img = np.asarray(img)[:,:,0]
    img = np.expand_dims(img, axis=0)
    
    def test():
        a = np.zeros((64, 64, 3))
        for k in range(0, 3):
            for i in range(0, 64):
                for j in range(0, 64):
                    if k == 0:
                        a[i][j][k] = (i / 63.0) * (j / 63.0)
                    elif k == 1:
                        a[i][j][k] = ((63.0 - i) / 63.0) * (j / 63.0)
                    else:
                        a[i][j][k] = (i / 63.0) * ((63.0 - j) / 63.0)
        a = np.expand_dims(a, 0)
        return a
    
    # get weights
    weights = conv_mixer_model.get_weights()
    weights_names = [weight.name for layer in conv_mixer_model.layers for weight in layer.weights]
    
    # get outputs
    outputs = [K.function([conv_mixer_model.input], [layer.output])([img]) for layer in conv_mixer_model.layers]
    outputs_names = [layer.name for layer in conv_mixer_model.layers]
    
    # dest = './model/jp_convmixer.bytes'
    
    # try:
    #     os.remove(dest)
    # except OSError:
    #     pass
    # for i in range(0, len(weights)):
    #     write_weights(weights[i], dest)