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
from imblearn.under_sampling import RandomUnderSampler

from modAL.models import ActiveLearner, Committee
from modAL.disagreement import vote_entropy_sampling
from modAL.batch import uncertainty_batch_sampling

from model import Classifier, create_model, get_model

SEED = 42
INITIAL_SIZE = 50
BATCH_SIZE = 10
# QUERY_SIZE = 20
N_QUERIES = 100
N_COMMITTEES = 10
EPOCHS = 20
LOSS = "sparse_categorical_crossentropy"
CALLBACK = EarlyStopping(monitor="loss", patience=2)
ESTIMATOR = KerasClassifier(build_fn=get_model, 
                            loss=LOSS, 
                            epochs=EPOCHS, 
                            batch_size=BATCH_SIZE,
                            #callbacks=[CALLBACK]
                            )
SAVEPATH_QUERY = "query_history_100.npy"
SAVEPATH_PERF = "performance_history_100.npy"
# ESTIMATOR = SGDClassifier(random_state=SEED, max_iter=1000, tol=1e-3)

### Data 
X_raw = np.load("image_data_gray.npy")
y_raw = np.load("labels.npy")

rus = RandomUnderSampler(random_state=0)
X_resampled, y_resampled = rus.fit_resample(X_raw.reshape(4000,-1), y_raw.reshape(4000,-1))
X_resampled, y_resampled = np.reshape(X_resampled, (2000,80,80,1)), np.reshape(y_resampled, (2000,)) 
#X_train, X_test, y_train, y_test = train_test_split(X_raw, y_raw, test_size=0.20, random_state=42)
X_pool, X_test, y_pool, y_test = train_test_split(X_resampled, y_resampled, test_size=0.50, random_state=42)
#X_pool, X_test, y_pool, y_test = train_test_split(X_raw, y_raw, test_size=0.50, random_state=42)

#X_train, X_pool, y_train, y_pool = train_test_split(X_train, y_train, train_size=INITIAL_SIZE, random_state=42)

# Initialize PCA for later projection
# X_flat = np.array([img.flatten() for img in X])
X_flat = np.array([img.flatten() for img in X_resampled])
pca = PCA(n_components=2, random_state=SEED)
transformed_X = pca.fit(X=X_flat)

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
performance_history = [unqueried_score]
query_history = []


### Active Loop
for idx in range(N_QUERIES):
    query_idx, query_instance = committee.query(X_pool)

    # Get position of query instance
    query_transformed = pca.transform(query_instance.flatten().reshape(1,-1))
    query_history.append(query_transformed)

    # Teach committee
    committee.teach(X=X_pool[query_idx],
                    y=y_pool[query_idx])
    performance_history.append(committee.score(X_test, y_test))

    # remove queried instance from pool
    X_pool = np.delete(X_pool, query_idx, axis=0)
    y_pool = np.delete(y_pool, query_idx)
    print(f"Query Index: {query_idx}")

np.save(SAVEPATH_QUERY, np.array(query_history))
np.save(SAVEPATH_PERF, np.array(performance_history))

print(performance_history)
# plt.plot(performance_history)
# plt.show()