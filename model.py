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