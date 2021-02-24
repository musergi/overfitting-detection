import os
import tensorflow as tf
import pandas as pd

class DatasetGatherer(tf.keras.callbacks.Callback):
    def __init__(self, folder):
        super(DatasetGatherer, self).__init__()
        self.history = {}
        self.index_counter = 0
        self.architecture = {}
        self.folder = folder

    def on_epoch_end(self, epoch, logs=None):
        # Save weights
        filepath = f'{self.index_counter}.tf'
        self.index_counter += 1
        self.model.save(os.path.join(self.folder, filepath), save_format='tf')

        # Save metrics
        logs = logs or {}
        logs['epoch'] = epoch
        logs['network_path'] = filepath
        logs.update(self.architecture)
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

    def save(self):
        pd.DataFrame(self.history).to_csv(os.path.join(self.folder, 'index.csv'))


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

dataset_gatherer = DatasetGatherer('dataset')
for layer_size in [64, 128, 256, 512, 1024]:
    dataset_gatherer.architecture['layer_size'] = layer_size
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=x_train.shape[1:]),
        tf.keras.layers.Dense(layer_size, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer='adam',
                  loss=loss_fn,
                  metrics=['accuracy'])
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, callbacks=[dataset_gatherer], verbose=0)
    print(f'Trained model {layer_size}')
dataset_gatherer.save()