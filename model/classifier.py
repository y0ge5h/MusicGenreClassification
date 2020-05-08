import json
import numpy as np
from sklearn.model_selection import train_test_split as tts
import tensorflow.keras as keras
import matplotlib.pyplot as plt

DATASET_PATH = 'dataset/data.json'


# load data
def load_data(dataset_path):
    with open(dataset_path, 'r') as jf:
        data = json.load(jf)

    # convert list into np arrays
    inputs = np.array(data['mfcc'])
    targets = np.array(data['label'])

    print(f"mfcc label {inputs.shape}")
    print(f"targets label {targets.shape}")
    return inputs, targets


def plot_history(history):
    fig, axs = plt.subplots(2)

    # accuracy subplots
    axs[0].plot(history.history['accuracy'],  label='train_accuracy')
    axs[0].plot(history.history['val_accuracy'], label='val_accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].legend(loc='lower right')
    axs[0].set_title('Accuracy Eval')

    # error subplots
    axs[1].plot(history.history['loss'], label='train_loss')
    axs[1].plot(history.history['val_loss'], label='val_loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].legend(loc='upper right')
    axs[1].set_title('Loss Eval')

    plt.show()


if __name__ == '__main__':
    inputs, targets = load_data(DATASET_PATH)
    print(f"{inputs.shape}, {targets.shape}")

    # split data
    X_train, X_test, y_train, y_test = tts(inputs, targets, test_size=0.3)
    print(f"X_train {X_train.shape}")
    print(f"y_train {y_train.shape}")
    print(f"X_test {X_test.shape}")
    print(f"y_test {y_test.shape}")
    exit(0)

    # build network using tf and keras > input , 3 hidden layers, output layer
    model = keras.Sequential([
        # input layer
        keras.layers.Flatten(input_shape=(X_train.shape[1], X_train.shape[2])),

        # hidden layer l2 regularization and dropout layers to avoid overfitting
        keras.layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),

        # output layer
        keras.layers.Dense(10, activation='softmax')
    ])

    # compile
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # train
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=52, batch_size=32)

    # plot accuracy and errors over epochs
    plot_history(history)
