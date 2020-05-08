import json
import numpy as np
from sklearn.model_selection import train_test_split as tts
import tensorflow.keras as keras

DATASET_PATH = 'dataset/data.json'


# load data
def load_data(dataset_path):
    with open(dataset_path, 'r') as jf:
        data = json.load(jf)

    # convert list into np arrays
    inputs = np.array(data['mfcc'])
    targets = np.array(data['label'])

    return inputs, targets


def build_model(input_shape):
    # create a model
    model = keras.Sequential()

    # 1st conv layer
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())  # process that standardizes the activations in current layer

    # 2nd conv layer
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # 3rd conv layer
    model.add(keras.layers.Conv2D(32, (2, 2), activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # flatten the output
    model.add(keras.layers.Flatten())

    # Dense layer
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.3))

    # output layer softmax
    model.add(keras.layers.Dense(10, activation='softmax'))

    print(model.summary())
    return model


def predict(X, y, model):
    X = X[np.newaxis, ...]
    prediction = model.predict(X)

    # extract index with max value
    prediction_index = np.argmax(prediction, axis=1)

    print(f"prediction {prediction_index[0]}, expected index {y}")


if __name__ == '__main__':
    # create train , validation , test sets
    X, y = load_data(DATASET_PATH)
    X_train, X_test, y_train, y_test = tts(X, y, test_size=0.25)
    X_train, X_val, y_train, y_val = tts(X_train, y_train, test_size=0.2)

    X_train = X_train[..., np.newaxis]
    X_val = X_val[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    print(f"{X_train},{X_val},{X_test}")

    # build the cnn network
    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
    model = build_model(input_shape=input_shape)

    # compile the network
    optimizer = keras.optimizers.Adam(0.0001)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # train the cnn
    model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=32, epochs=40)

    # evaluate on test
    test_error, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
    print(f"accuracy on test set {test_accuracy}")

    # make predictions on sample
    X = X_test[100]
    y = y_test[100]
    print(X.shape, y.shape)
    predict(X, y, model)
