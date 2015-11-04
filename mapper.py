#!/local/anaconda/bin/python
# IMPORTANT: leave the above line as is.
import sys
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.kernel_approximation import (RBFSampler,Nystroem)

from sklearn.svm import SVC

DIMENSION = 400  # Dimension of the original data.
CLASSES = (-1, +1)   # The classes that we are trying to predict.


np.random.seed(seed=42)

def transform(x_original):
    #w_feature = RBFSampler(gamma=1, random_state=1)
    w_feature = Nystroem(kernel='rbf', gamma=None, coef0=1, degree=3, kernel_params=None, n_components=100, random_state=None)
    x = w_feature.fit_transform(x_original)
    return x

if __name__ == "__main__":
    #f = open('/Users/Hangxin/Desktop/training_set.txt')
    clf = SGDClassifier(loss='hinge', penalty='l2',alpha=0.0001, l1_ratio=0.15, fit_intercept=False, n_iter=1, shuffle=True, verbose=0, epsilon=0.1, n_jobs=1, random_state=None, learning_rate='optimal', eta0=0.0, power_t=0.5, class_weight=None, warm_start=False, average=False)
    #Extract Features
    index = 0
    for line in sys.stdin:
        line = line.strip()
        label, x_string = line.split(" ", 1)
        label = int(label)
        x_original = np.fromstring(x_string, sep=' ')
        x = transform(x_original) #using the kernel function  
        clf.partial_fit(x, [label], CLASSES)

    for x in clf.coef_[0]:
        print x,
#    print

#cat training_set.txt | python mapper.py | python reducer.py > r_weights.txt
#python evaluate.py r_weights.txt test_data.txt  test_label.txt /Users/Charles/Desktop/

