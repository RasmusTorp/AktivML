from tqdm import tqdm
# from model import get_model
import numpy as np
from collections import namedtuple
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Activation
# from keras.wrappers.scikit_learn import KerasClassifier
from scikeras.wrappers import KerasClassifier
from modAL.models import ActiveLearner
from sklearn.model_selection import train_test_split
from model import create_model, get_model
import pandas as pd

INITIAL_SIZE = 10


n_repeats = 2
n_queries = 10
SEED = 42
TEST_SIZE = 1/5
BATCH_SIZE = 2
EPOCHS = 10
verbose = 1


ResultsRecord = namedtuple('ResultsRecord', ['estimator', 'query_id', 'score'])

X = np.load("image_data_gray.npy")
y = np.load("labels.npy")

SHAPE = X[0].shape

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED)
meta = {"n_features_in_": 0
        ,"X_shape_": (80,80,1)
        ,"n_classes_": 2}


# def get_model(meta):
#     # note that meta is a special argument that will be
#     # handed a dict containing input metadata
#     n_features_in_ = meta["n_features_in_"]
#     X_shape_ = meta["X_shape_"]
#     n_classes_ = meta["n_classes_"]

#     model = Sequential()
#     model.add(Conv2D(80,(3,3), activation='relu', input_shape=X_shape_))
#     model.add(MaxPooling2D((2,2), padding='same'))
#     model.add(Conv2D(40,(3,3), activation='relu'))
#     model.add(MaxPooling2D((2,2), padding='same'))
#     model.add(Flatten())
#     model.add(Dense(40))
#     model.add(Activation("relu"))
#     model.add(Dense(n_classes_))
#     model.add(Activation("softmax"))
#     return model

# def get_scikit_model():
#     return KerasClassifier(create_model("sparse_categorical_cross_entropy","adam","relu",SHAPE))

def get_scikit_model():
    return KerasClassifier(get_model,  
                           epochs=EPOCHS,
                           optimizer='adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])




# def get_learner():
#     learner = ActiveLearner(estimator=KerasClassifier(build_fn=get_model, 
#                                                   loss="sparse_categorical_crossentropy", 
#                                                   epochs=3, 
#                                                   batch_size=64), 
#                         X_training=X_train, 
#                         y_training=y_train)
#     return


# n_labeled_examples = X.shape[0]
# training_indices = np.random.randint(low=0, high=n_labeled_examples + 1, size=INITIAL_SIZE)
# X_train = X[training_indices]
# y_train = y[training_indices]


# model = get_model(meta)

permutations = [np.random.permutation(X_train.shape[0]) for _ in range(n_repeats)]

random_results = []

# for i_repeat in tqdm(range(n_repeats)):
#     model = create_model("sparse_categorical_crossentropy","adam","relu",SHAPE)
#     for i_query in tqdm(range(1,n_queries),leave=False):
#         query_indices=permutations[i_repeat][:1+i_query]

#         model.fit(X_train[query_indices, :],y_train[query_indices],batch_size=BATCH_SIZE,
#                               epochs=epochs,validation_data=(X_test,y_test),verbose=verbose)

#         loss,acc = model.evaluate(X_test,y_test)


#         # learner=learner.fit(x=X_train[query_indices, :], y=y_train[query_indices])
#         # score = learner.score(X_test, y_test)
        
#         random_results.append(ResultsRecord('random', i_query, acc))


random_results = []

# learner = get_scikit_model()
# learner.fit(X_train,y_train)
# score = learner.score(X_test,y_test)
# print(score)

for i_repeat in tqdm(range(n_repeats)):
    learner = get_scikit_model()
    for i_query in tqdm(range(1,n_queries),leave=False):
        query_indices=permutations[i_repeat][:1+i_query]
        learner=learner.fit(X=X_train[query_indices, :], y=y_train[query_indices])
        score = learner.score(X_test, y_test)
        
        random_results.append(ResultsRecord('random', i_query, score))


print(pd.DataFrame(results) for results in random_results)