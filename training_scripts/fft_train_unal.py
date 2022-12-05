from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pandas as pd
import os

from helpers import UnalRegressionSequence, Rotate90Randomly, Fourier2D, Scheduler


# Create a list of epochs
name = 'unal_mm_abs_resized_256'
epochs = 500
batch_size = 64
val_batch_size = 64
changing_epoch = 450
changing_period = 100
learning_rate = 0.001
decay = 0.1
factor = 1
dropout = 0.2
title = f'{name}_e{epochs}_ce{changing_epoch}_lr{learning_rate}_d{decay}_bs{batch_size}_' \
        f'dr{dropout}_cp_{changing_period}_x{factor}'
data_dir = '../dataset_unal'
holo_dir = f'{data_dir}/Hologramas'

df = pd.read_csv(f'{data_dir}/database_unal.csv')
df.columns = ['sample', 'name', 'reconstruction', 'axial', 'transverse', 'distance', 'l',
              'wavelength', 'path_holo', 'path_r']

gpus = tf.config.list_physical_devices('GPU')
logical_gpus = tf.config.list_logical_devices('GPU')
print(len(gpus), 'Physical GPUs,', len(logical_gpus), 'Logical GPU')

x_set = [x for x in os.listdir(holo_dir) if '.png' in x]

y_set = pd.DataFrame()
for name in (names.split('.')[0] for names in x_set if '.png' in names):
    y_set = pd.concat([y_set, df[df['name'] == name]['distance'] * 1000])
    # y_set = pd.concat([y_set, df[df['name'] == name]['distance']])
y_set = y_set[0].tolist()


x_train, x_test, y_train, y_test = train_test_split(x_set, y_set, test_size=0.2, random_state=42)


gpus = tf.config.list_logical_devices('GPU')
strategy = tf.distribute.MirroredStrategy(gpus)
img_height = 1024
img_width = 1024

sequence_train = UnalRegressionSequence(x_train, y_train, holo_dir, batch_size)
sequence_val = UnalRegressionSequence(x_test, y_test, holo_dir, val_batch_size)

with strategy.scope():
    # Model creation
    model = Sequential([
        # It is very important that the dtype is complex64 instead of complex128 to avoid nan problems
        tf.keras.layers.Rescaling(1. / 255, input_shape=(1024, 1024, 1), dtype=tf.complex64),
        # tf.keras.layers.Rescaling(1. / 255, input_shape=(1024, 1024, 1), dtype=tf.float64),
        tf.keras.layers.Resizing(256, 256, dtype=tf.complex64),
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
        # layers.Conv2D(256 * factor, 3, padding='same', activation='swish'),
        # layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Dropout(dropout),
        layers.Flatten(),
        layers.Dense(32 * factor, activation='swish'),
        # layers.Dropout(dropout),
        # layers.Dense(32 * factor, activation='swish'),
        layers.Dense(1)
    ])
    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate), metrics=['mae'])


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
