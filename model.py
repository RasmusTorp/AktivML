import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.wrappers.scikit_learn import KerasClassifier


def create_model(loss_func, opt, act, input_shape):
    model = Sequential()
    model.add(Conv2D(80,(3,3),activation=act,input_shape=input_shape))
    model.add(MaxPooling2D((2,2), padding='same'))
    # model.add(Dropout(dropRate1))
    model.add(Conv2D(40,(3,3),activation=act))
    model.add(MaxPooling2D((2,2), padding='same'))
    model.add(Flatten())
    model.add(Dense(40,activation=act))
    # model.add(Dropout(dropRate2))
    model.add(Dense(2,activation="softmax"))
    model.compile(optimizer=opt,loss=loss_func,metrics=["accuracy"])

    return model

SHAPE = (80,80,1)
Classifier = KerasClassifier(create_model("sparse_categorical_cross_entropy","adam","relu",SHAPE))


# Compatibel with the scikeras library
def get_model(meta):
    # note that meta is a special argument that will be
    # handed a dict containing input metadata
    n_features_in_ = meta["n_features_in_"]
    X_shape_ = meta["X_shape_"]
    n_classes_ = meta["n_classes_"]

    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(80,(3,3), activation='relu', input_shape=X_shape_[1:]))
    model.add(keras.layers.MaxPooling2D((2,2), padding='same'))
    model.add(keras.layers.Conv2D(40,(3,3), activation='relu'))
    model.add(keras.layers.MaxPooling2D((2,2), padding='same'))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(40))
    model.add(keras.layers.Activation("relu"))
    model.add(keras.layers.Dense(n_classes_))
    model.add(keras.layers.Activation("softmax"))
    return model
