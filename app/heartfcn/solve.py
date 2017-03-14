import caffe
#import surgery, score

import numpy as np
import os
import sys

import surgery, score

try:
    import setproctitle
    setproctitle.setproctitle(os.path.basename(os.getcwd()))
except:
    pass

# weights = 'snapshot/train_iter_1000.caffemodel'

# init
if '-gpu' in sys.argv:
    caffe.set_mode_gpu()
    caffe.set_device(0)
else:
    caffe.set_mode_cpu()

solver = caffe.SGDSolver('solver.prototxt')

# surgeries
interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
surgery.interp(solver.net, interp_layers)

# scoring
val = np.loadtxt('../../data/segvalid11.txt', dtype=str)

for _ in range(25):
    solver.step(4000)

    score.seg_tests(solver, False, val, layer='score_output2')
