# data_proj2
**Introduction**

data mining project 2 implemented with SGD and SVM kernel
implemented with sklearn class RBFSampler,Nystroem,AdditiveChi2Sampler,SkewedChi2Sampler
Final version with AdditiveChi2Sampler

**HOW TO RUN**

1. download repository to your local drive and cd to the folder data_proj2.  
2. Copy and paste you previous data files to the folder: training_set.txt  test_data.txt  test_label.txt
3. To compute weights: cat training_set.txt | python mapper.py | python reducer > weights.txt
4. As an exmaple, the computed weights is already in the folder: weights_4.txt
5. To compute accuracy: evaluate.py weights_4.txt test_data.txt test_label.txt <your repository folder>


**RESULT**
- The current method AdditiveChi2Sampler have accuracy = 0.80463
- SkewedChi2Sampler(skewedness=1.0, n_components=100, random_state=1) accuracy  = 0.703116
- tune gamma and n_components of RBFSampler to get better accuracy
  - when RBFSampler(gamma=0.1, n_components=100), accuracy = 0.687130
  - when RBFSampler(gamma=0.1, n_components=200), accuracy = 0.681268
  - when RBFSampler(gamma=1, n_components= 100), accuracy = 0.714028
- without any kernel transform accuracy = 0.74747
