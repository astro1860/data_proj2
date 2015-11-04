#!/local/anaconda/bin/python
# IMPORTANT: leave the above line as is.

import logging
import sys
import numpy as np

data_w = np.array([]).reshape(0,200)
for line in sys.stdin:
    line = line.strip()
    data_w = np.vstack((data_w, np.asarray(np.fromstring(line, sep=' '))))
data_w = np.mean(data_w, axis = 0)

for x in data_w:
	print x,