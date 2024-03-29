import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, accuracy_score
import matplotlib as mpl
import matplotlib.pyplot as plt

from functools import partial
from modAL.batch import uncertainty_batch_sampling
from modAL.models import ActiveLearner
#from keras.wrappers.scikit_learn import KerasClassifier
from scikeras.wrappers import KerasClassifier
from imblearn.under_sampling import RandomUnderSampler

from model import Classifier, create_model, get_model

SEED = 42
INITIAL_SIZE = 50
BATCH_SIZE = 10
QUERY_SIZE = 5
N_QUERIES = 10
EPOCHS = 20
SAVEPATH_QUERY = "query_history_rank_50.npy"
SAVEPATH_PERF = "performance_history_rank_50.npy"
SAVEPATH_RANK_IMG = "top10_ranked_images_50.npy"
SAVEPATH_RANK_IMG_LABEL = "top10_ranked_images_labels_50.npy"

### Data 
X_raw = np.load("image_data_gray.npy")
y_raw = np.load("labels.npy")
rus = RandomUnderSampler(random_state=0)
X_resampled, y_resampled = rus.fit_resample(X_raw.reshape(4000,-1), y_raw.reshape(4000,-1))
X_resampled, y_resampled = np.reshape(X_resampled, (2000,80,80,1)), np.reshape(y_resampled, (2000,)) 

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.50, random_state=42)
X_train, X_pool, y_train, y_pool = train_test_split(X_train, y_train, train_size=INITIAL_SIZE, random_state=42)


# Initialize PCA for later projection
# X_flat = np.array([img.flatten() for img in X])
X_flat = np.array([img.flatten() for img in X_resampled])
pca = PCA(n_components=2, random_state=SEED)
transformed_X = pca.fit(X=X_flat)


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
preset_batch = partial(uncertainty_batch_sampling, n_instances=QUERY_SIZE)
learner = ActiveLearner(estimator=KerasClassifier(build_fn=get_model, 
                                                  loss="sparse_categorical_crossentropy", 
                                                  epochs=EPOCHS, 
                                                  batch_size=BATCH_SIZE), 
                        X_training=X_train, 
                        y_training=y_train,
                        query_strategy=preset_batch)

### Initial prediction after 1 run
# predictions = learner.predict(X_test)
# is_correct = (predictions == y_test)
# print(is_correct[:50])



### Loop .....
#unqueried_score = learner.score(X_raw, y_raw)


performance_history = []
query_history = []
top_rank_imgs = []
top_rank_imgs_labels = []

for index in range(N_QUERIES):
  query_index, query_instance = learner.query(X_pool)

  # Get position of query instance
  query_transformed = [pca.transform(instance.flatten().reshape(1,-1)) for instance in query_instance]
  query_history.append(query_transformed)

  # Save highest ranked images
  top_rank_imgs.append(query_instance[:10])
  top_rank_imgs_labels.append(y_pool[query_index])
  
  # Teach our ActiveLearner model the record it has requested.
  X, y = X_pool[query_index], y_pool[query_index]
  learner.teach(X=X, y=y)

  # Remove the queried instance from the unlabeled pool.
  X_pool, y_pool = np.delete(X_pool, query_index, axis=0), np.delete(y_pool, query_index)

  # Calculate and report our model's accuracy.
  preds = learner.predict(X_test)
  # model_accuracy = learner.score(X_test, y_raw)
  model_accuracy = f1_score(y_test, preds)
  # print('Accuracy after query {n}: {acc:0.4f}'.format(n=index + 1, acc=model_accuracy))

  # Save our model's performance for plotting.
  performance_history.append(model_accuracy)

print(performance_history)

np.save(SAVEPATH_QUERY, np.array(query_history))
np.save(SAVEPATH_PERF, np.array(performance_history))
np.save(SAVEPATH_RANK_IMG, np.array(top_rank_imgs[0]))
np.save(SAVEPATH_RANK_IMG_LABEL, top_rank_imgs_labels[0])

print()
# plt.plot(performance_history)
# plt.show()