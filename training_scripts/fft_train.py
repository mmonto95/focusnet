import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pandas as pd
import os

from helpers import RegressionSequence, Rotate90Randomly, Fourier2D, Scheduler


# Create a list of epochs
name = '2headed_less_layers'
epochs = 150
batch_size = 16
val_batch_size = 32
changing_epoch = 120
changing_period = 30
learning_rate = 0.001
decay = 0.1
factor = 2
dropout = 0.25
title = f'{name}_e{epochs}_ce{changing_epoch}_lr{learning_rate}_d{decay}_bs{batch_size}_' \
        f'dr{dropout}_cp_{changing_period}_x{factor}'
data_dir = '../holors'

# noinspection PyArgumentList
df = pd.read_csv(f'{data_dir}/holors_total.csv')

# tf.debugging.set_log_device_placement(True)
gpus = tf.config.list_physical_devices('GPU')
logical_gpus = tf.config.list_logical_devices('GPU')
print(len(gpus), 'Physical GPUs,', len(logical_gpus), 'Logical GPU')

x_set = os.listdir(f'{data_dir}/D1') + os.listdir(f'{data_dir}/D2') + os.listdir(f'{data_dir}/D3') + os.listdir(
    f'{data_dir}/D4') + os.listdir(f'{data_dir}/D5')
# x_set = x_set[:100]

y_set = pd.DataFrame()
for name in (names.split('.')[0] for names in x_set):
    y_set = pd.concat([y_set, df[df['name'] == name]['distance'] * 100])
    # y_set = pd.concat([y_set, df[df['name'] == name]['distance']])
y_set = y_set[0].tolist()

# y_set = []
# for name in (names.split('.')[0] for names in x_set):
#     y_set.append(df[df['name'] == name]['distance'].values[0])
# y_set = tf.keras.utils.normalize(np.array(y_set))

x_train, x_test, y_train, y_test = train_test_split(x_set, y_set, test_size=0.2, random_state=42)

# x_train = x_train[:65] + x_train[75:]
# y_train = y_train[:65] + y_train[75:]


gpus = tf.config.list_logical_devices('GPU')
strategy = tf.distribute.MirroredStrategy(gpus)
img_height = 1024
img_width = 1024

sequence_train = RegressionSequence(x_train, y_train, data_dir, batch_size)
sequence_val = RegressionSequence(x_test, y_test, data_dir, val_batch_size)

with strategy.scope():
    # Model creation
    model = Sequential([
        # It is very important that the dtype is complex64 instead of complex128 to avoid nan problems
        tf.keras.layers.Rescaling(1. / 255, input_shape=(1024, 1024, 1), dtype=tf.complex64),
        # tf.keras.layers.Rescaling(1. / 255, input_shape=(1024, 1024, 1), dtype=tf.float64),
        Rotate90Randomly(),
        Fourier2D(),
        layers.Conv2D(8 * factor, 7, padding='same', activation='swish'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Conv2D(16 * factor, 5, padding='same', activation='swish'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Conv2D(32 * factor, 5, padding='same', activation='swish'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Conv2D(64 * factor, 3, padding='same', activation='swish'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        # layers.Conv2D(128 * factor, 3, padding='same', activation='swish'),
        # layers.BatchNormalization(),
        # layers.MaxPooling2D(),
        layers.Dropout(dropout),
        layers.Flatten(),
        layers.Dense(32 * factor, activation='swish'),
        layers.Dropout(dropout),
        layers.Dense(32 * factor, activation='swish'),
        layers.Dense(1)
    ])
    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate), metrics=['mae'])
    # model.compile(loss='mse', optimizer='sgd', metrics=['mae'])
    # model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate, clipnorm=1), metrics=['mae'])
    # model.compile(loss='mse', optimizer=tf.keras.optimizers.RMSprop(learning_rate), metrics=['mae'])


scheduler = Scheduler(changing_period, changing_epoch)

checkpoint_path = f'../models/{title}_cp'
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 verbose=1,
                                                 save_best_only=True)


history = model.fit(sequence_train, epochs=epochs, validation_data=sequence_val,
                    # callbacks=[history_logger, cp_callback])
                    callbacks=[tf.keras.callbacks.LearningRateScheduler(scheduler.schedule),
                               cp_callback])

os.makedirs(f'../models/{title}', exist_ok=True)
model.save(f'../models/{title}')

hist_folder = '../hist'
os.makedirs(hist_folder, exist_ok=True)
hist_csv_file = f'{hist_folder}/{title}.csv'
hist_df = pd.DataFrame(history.history)
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)
