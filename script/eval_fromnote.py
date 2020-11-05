from os.path import join,exists,realpath,dirname,basename
from os import makedirs,listdir, system
import numpy as np, _pickle as cPickle, editdistance, seaborn as sns
import matplotlib.pyplot as plt, pandas as pd, itertools, glob, h5py
from scipy.stats import entropy
from matplotlib.font_manager import FontProperties
from IPython.display import display
from collections import defaultdict
from IPython.display import display
# from itertools import izip
from scipy.stats import ranksums
import multiprocessing as mp
from PIL import Image

import inception_score

rundir = 'cifar10/'
e = 100

def get_score(improved_keras_dir, t_n_epoch):
    score = []
    for i in range(t_n_epoch-10, t_n_epoch):
        print(i)
        # scorefile = join(improved_keras_dir, 'epoch_{}.score'.format(i))
        # if not exists(scorefile):   
        datafile = join(improved_keras_dir, 'epoch_{}.pkl'.format(i))
        if not exists(datafile):
            break
        with open(datafile, 'rb') as f:
            sample = cPickle.load(f)
            print(len(list(sample)))
            t_score = inception_score.get_inception_score(list(sample), 1)[0]
        # with open(scorefile, 'w') as f:
        #     f.write('%f\n' % t_score)l
        # else:
        #     with open(scorefile) as f:
        #         t_score = float(f.readline())
        score.append(t_score)
    
    return max(score)

expt2plot = ['adam_ratio1', 'optimAdam', 'optimAdam_ratio1']

for expt in expt2plot:
    score = get_score(join(rundir, expt), e)
    print(expt, score)
