import argparse
import os
import pathlib
import random
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense, Dropout, Conv2D, MaxPooling2D,
                                     Flatten)

print('Using Tensorflow version:', tf.__version__)

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--cpus', type=int, default=10)
parser.add_argument('--datadir', type=str, default=None)
args = parser.parse_args()

BATCH_SIZE = 32
INPUT_IMAGE_SIZE = [160, 160, 3]

if args.datadir is None:
    DATADIR = "/scratch/dac/data/"
else:
    DATADIR = args.datadir

print('Using DATADIR', DATADIR)
datapath = os.path.join(DATADIR, "dogs-vs-cats/train-2000/")

print('Using {} CPUs'.format(args.cpus))

nimages = dict()
nimages['train'] = 2000
nimages['validation'] = 1000


def get_paths(dataset):
    data_root = pathlib.Path(datapath+dataset)
    image_paths = list(data_root.glob('*/*'))
    image_paths = [str(path) for path in image_paths]
    image_count = len(image_paths)
    assert image_count == nimages[dataset], \
        "Found {} images, expected {}".format(image_count, nimages[dataset])
    return image_paths


image_paths = dict()
image_paths['train'] = get_paths('train')
image_paths['validation'] = get_paths('validation')

label_names = sorted(item.name for item in
                     pathlib.Path(datapath+'train').glob('*/')
                     if item.is_dir())
label_to_index = dict((name, index) for index, name in enumerate(label_names))


def get_labels(dataset):
    return [label_to_index[pathlib.Path(path).parent.name]
            for path in image_paths[dataset]]


image_labels = dict()
image_labels['train'] = get_labels('train')
image_labels['validation'] = get_labels('validation')


def preprocess_image(image, augment):
    image = tf.image.decode_jpeg(image, channels=3)
    if augment:
        image = tf.image.resize(image, [256, 256])
        image = tf.image.random_crop(image, INPUT_IMAGE_SIZE)
        if random.random() < 0.5:
            image = tf.image.flip_left_right(image)
    else:
        image = tf.image.resize(image, INPUT_IMAGE_SIZE[:2])
    image /= 255.0  # normalize to [0,1] range
    return image


def load_and_augment_image(path, label):
    image = tf.io.read_file(path)
    return preprocess_image(image, True), label


def load_and_not_augment_image(path, label):
    image = tf.io.read_file(path)
    return preprocess_image(image, False), label


train_dataset = tf.data.Dataset.from_tensor_slices((image_paths['train'],
                                                    image_labels['train']))
validation_dataset = tf.data.Dataset.from_tensor_slices((image_paths['validation'],
                                                         image_labels['validation']))

train_dataset = train_dataset.map(load_and_augment_image, num_parallel_calls=args.cpus)
train_dataset = train_dataset.shuffle(2000).batch(BATCH_SIZE, drop_remainder=True)
train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

validation_dataset = validation_dataset.map(load_and_not_augment_image,
                                            num_parallel_calls=args.cpus)
validation_dataset = validation_dataset.batch(BATCH_SIZE, drop_remainder=True)
validation_dataset = validation_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=INPUT_IMAGE_SIZE, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(train_dataset, epochs=args.epochs,
          validation_data=validation_dataset,
          verbose=2)

# fname = "dvc-cnn-simple.h5"
# print('Saving model to', fname)
# model.save(fname)
print('DONE!')
