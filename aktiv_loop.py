import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import matplotlib as mpl
import matplotlib.pyplot as plt

from functools import partial
from modAL.batch import uncertainty_batch_sampling
from modAL.models import ActiveLearner
#from keras.wrappers.scikit_learn import KerasClassifier
from scikeras.wrappers import KerasClassifier

from model import Classifier, create_model, get_model

#RANDOM_STATE_SEED = 42
INITIAL_SIZE = 3
#SHAPE = (80,80,1)
BATCH_SIZE = 1

### Data 
X_raw = np.load("image_data_gray.npy")
y_raw = np.load("labels.npy")
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Isolate our examples for our labeled dataset.
n_labeled_examples = X_raw.shape[0]
training_indices = np.random.randint(low=0, high=n_labeled_examples + 1, size=INITIAL_SIZE)
X_train = X_raw[training_indices]
y_train = y_raw[training_indices]

# Isolate the non-training examples we'll be querying.
X_pool = np.delete(X_raw, training_indices, axis=0)
y_pool = np.delete(y_raw, training_indices, axis=0)


### Active learner

# Pool based
# See https://modal-python.readthedocs.io/en/latest/content/examples/pool-based_sampling.html
learner = ActiveLearner(estimator=KerasClassifier(build_fn=get_model, 
                                                  loss="sparse_categorical_crossentropy", 
                                                  epochs=3, 
                                                  batch_size=64), 
                        X_training=X_train, 
                        y_training=y_train)

# (Alternative) Ranked batch sampling
# See https://modal-python.readthedocs.io/en/latest/content/examples/ranked_batch_mode.html
#preset_batch = partial(uncertainty_batch_sampling, n_instances=BATCH_SIZE)
#learner = ActiveLearner(estimator=KerasClassifier(build_fn=get_model, 
#                                                   loss="sparse_categorical_crossentropy", 
#                                                   epochs=3, 
#                                                   batch_size=64), 
#                         X_training=X_train, 
#                         y_training=y_train,
#                         query_strategy=preset_batch)

### Initial prediction after 1 run
predictions = learner.predict(X_raw)
is_correct = (predictions == y_raw)
print(is_correct[:50])



### Loop .....
#unqueried_score = learner.score(X_raw, y_raw)

N_QUERIES = 20
performance_history = []

for index in range(N_QUERIES):
  query_index, query_instance = learner.query(X_pool)

  # Teach our ActiveLearner model the record it has requested.
  X, y = X_pool[query_index], y_pool[query_index]
  learner.teach(X=X, y=y)

  # Remove the queried instance from the unlabeled pool.
  X_pool, y_pool = np.delete(X_pool, query_index, axis=0), np.delete(y_pool, query_index)

  # Calculate and report our model's accuracy.
  model_accuracy = learner.score(X_raw, y_raw)
  print('Accuracy after query {n}: {acc:0.4f}'.format(n=index + 1, acc=model_accuracy))

  # Save our model's performance for plotting.
  performance_history.append(model_accuracy)

print(performance_history)