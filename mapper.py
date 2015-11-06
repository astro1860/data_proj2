#!/local/anaconda/bin/python
# IMPORTANT: leave the above line as is.
import sys
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.kernel_approximation import (RBFSampler,Nystroem,AdditiveChi2Sampler,SkewedChi2Sampler)

from sklearn.svm import SVC

#DIMENSION = 400  # Dimension of the original data.
CLASSES = (-1, +1)   # The classes that we are trying to predict.


np.random.seed(seed=42)

def transform(x_original):
    #w_feature = RBFSampler(gamma=1, random_state=1)
    #w_feature = Nystroem(kernel='rbf', gamma=1.0, n_components=400, random_state=1)
    w_feature = AdditiveChi2Sampler(sample_steps=2, sample_interval=None)
    #w_feature = SkewedChi2Sampler(skewedness=1.0, n_components=100, random_state=1)
    x = w_feature.fit_transform(x_original)
    return x

if __name__ == "__main__":
    #f = open('/Users/Hangxin/Desktop/training_set.txt')
    clf = SGDClassifier(loss='hinge', penalty='l1',alpha=0.0001/15, l1_ratio=0.15, fit_intercept=False, n_iter=1, shuffle=True, verbose=0, epsilon=0.1, n_jobs=1, random_state=1, learning_rate='optimal', eta0=0.0, power_t=0.5, class_weight=None, warm_start=False, average=True)
    #Extract Features
    #alpha = 0.0001 = a l1_ratio = 0.15
    #0.000001 --> 0.779509  a/100
    #0.00001 --> 0.813194   a/10
    #0.0001/20 --> 0.811584 a/20
    #0.0001/15 --> 0.813287 a/15 !!
    #0.0001/12-->0.807602 a/12
    #0.0001/17 --> 0.811538

    #0.814882 <-- L1_RATION 0.5 L1
    #0.819020 <-- l1 RATION 0.5 L1 average = true alpha = a/15 !!
    #0.817806 <-- l2 ration 0.15 average = true alpha = a/15
    #0.818444 <-- l2 ration 0.15 average  =true alpha = a/100
    #0.818413 <-- l2 ration 0.15 average = true alpha = a/50 0.819020
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

