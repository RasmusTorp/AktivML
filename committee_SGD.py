import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, accuracy_score
import matplotlib as mpl
import matplotlib.pyplot as plt
from functools import partial
#from keras.wrappers.scikit_learn import KerasClassifier
from scikeras.wrappers import KerasClassifier
from keras.callbacks import EarlyStopping

from modAL.models import ActiveLearner, Committee
from modAL.disagreement import vote_entropy_sampling
from modAL.batch import uncertainty_batch_sampling

from model import Classifier, create_model, get_model

SEED = 42
#SHAPE = (80,80,1)

INITIAL_SIZE = 50
BATCH_SIZE = 5
QUERY_SIZE = 20
N_QUERIES = 5
N_COMMITTEES = 4
EPOCHS = 10
LOSS = "sparse_categorical_crossentropy"
# CALLBACK = EarlyStopping(monitor="loss", patience=3)
# ESTIMATOR = KerasClassifier(build_fn=get_model, 
#                             loss=LOSS, 
#                             epochs=EPOCHS, 
#                             batch_size=BATCH_SIZE,
#                             callbacks=[CALLBACK])
ESTIMATOR = SGDClassifier(random_state=SEED, max_iter=1000, tol=1e-3)

### Data 
X_raw = np.load("image_data_gray.npy")
y_raw = np.load("labels.npy")
#X_train, X_test, y_train, y_test = train_test_split(X_raw, y_raw, test_size=0.20, random_state=42)
X_pool, X_test, y_pool, y_test = train_test_split(X_raw, y_raw, test_size=0.20, random_state=42)
#X_train, X_pool, y_train, y_pool = train_test_split(X_train, y_train, train_size=INITIAL_SIZE, random_state=42)


# # Isolate our examples for our labeled dataset.
# n_labeled_examples = X_train.shape[0]
# training_indices = np.random.randint(low=0, high=n_labeled_examples + 0, size=INITIAL_SIZE)
# X_trainAL = X_raw[training_indices]
# y_trainAL = y_raw[training_indices]

# # Isolate the non-training examples we'll be querying.
# X_pool = np.delete(X_train, training_indices, axis=0)
# y_pool = np.delete(y_train, training_indices, axis=0)


### Active learner

# Pool based
# See https://modal-python.readthedocs.io/en/latest/content/examples/pool-based_sampling.html
# learner = ActiveLearner(estimator=KerasClassifier(build_fn=get_model, 
#                                                   loss="sparse_categorical_crossentropy", 
#                                                   epochs=3, 
#                                                   batch_size=64), 
#                         X_training=X_train, 
#                         y_training=y_train)

# (Alternative) Ranked batch sampling
# See https://modal-python.readthedocs.io/en/latest/content/examples/ranked_batch_mode.html
# preset_batch = partial(uncertainty_batch_sampling, n_instances=QUERY_SIZE)
# learner = ActiveLearner(estimator=KerasClassifier(build_fn=get_model, 
#                                                   loss="sparse_categorical_crossentropy", 
#                                                   epochs=EPOCHS, 
#                                                   batch_size=BATCH_SIZE), 
#                         X_training=X_train, 
#                         y_training=y_train,
#                         query_strategy=preset_batch)

### Initial prediction after 1 run
# predictions = learner.predict(X_test)
# is_correct = (predictions == y_test)
# print(is_correct[:50])



### Loop .....
#unqueried_score = learner.score(X_raw, y_raw)


# initializing Committee members
learner_list = list()

for member_idx in range(N_COMMITTEES):
    # initial training data
    train_idx = np.random.choice(range(X_pool.shape[0]), size=INITIAL_SIZE, replace=False)
    X_train = X_pool[train_idx]
    y_train = y_pool[train_idx]

    # creating a reduced copy of the data with the known instances removed
    X_pool = np.delete(X_pool, train_idx, axis=0)
    y_pool = np.delete(y_pool, train_idx)

    # initializing learner
    learner = ActiveLearner(estimator = ESTIMATOR, 
                            X_training = X_train, 
                            y_training = y_train)
    learner_list.append(learner)

# assembling the committee
committee = Committee(learner_list=learner_list,
                      query_strategy=vote_entropy_sampling)

unqueried_score = committee.score(X_test, y_test)


### Active Loop
performance_history = [unqueried_score]
for idx in range(N_QUERIES):
    query_idx, query_instance = committee.query(X_pool)
    # committee.teach(X=X_pool[query_idx].reshape(1, -1),
    #                 y=y_pool[query_idx].reshape(1, ))
    committee.teach(X=X_pool[query_idx],
                    y=y_pool[query_idx])
    performance_history.append(committee.score(X_test, y_test))
    # remove queried instance from pool
    X_pool = np.delete(X_pool, query_idx, axis=0)
    y_pool = np.delete(y_pool, query_idx)
    print(f"Query Index: {query_idx}")

print(performance_history)

# performance_history = []
# for index in range(N_QUERIES):
#   query_index, query_instance = learner.query(X_pool)

#   # Teach our ActiveLearner model the record it has requested.
#   X, y = X_pool[query_index], y_pool[query_index]
#   learner.teach(X=X, y=y)

#   # Remove the queried instance from the unlabeled pool.
#   X_pool, y_pool = np.delete(X_pool, query_index, axis=0), np.delete(y_pool, query_index)

#   # Calculate and report our model's accuracy.
#   preds = learner.predict(X_test)
#   # model_accuracy = learner.score(X_test, y_raw)
#   model_accuracy = f1_score(y_test, preds)
#   print('Accuracy after query {n}: {acc:0.4f}'.format(n=index + 1, acc=model_accuracy))

#   # Save our model's performance for plotting.
#   performance_history.append(model_accuracy)

# print(performance_history)

plt.plot(performance_history)
plt.show()