#import matplotlib, numpy, scipy and other useful libs
from __future__ import division #otherwise 1/73 = 0 in python2
from numpy import *
import numpy as np
import scipy
import pylab
import matplotlib.pyplot as plt # plotting libraries from matlab
from scipy.stats import multivariate_normal
from dateutil import parser
import matplotlib.dates as md
import scipy.io as sio
from scipy.optimize import curve_fit # for fitting