#Function to initialize ipython notebook with matplotlib, seaborn, pandas, bokeh, scipy...
from __future__ import division #otherwise 1/73 = 0 in python2
from numpy import *
import scipy
import matplotlib.pyplot as plt # plotting libraries from matlab
from scipy.stats import multivariate_normal
from dateutil import parser
import matplotlib.dates as md
import scipy.io as sio
from scipy.optimize import curve_fit # for fitting
import pandas as pd

#define function to convert matfile into dataframe (panda)
def df_from_mat(mat_data):
    a=[]
    keys=[]
    
    for key in mat_data.keys():
        if key[0]!='_':
            a.append(mat_data[key])
            keys.append(key)
    keys=array(keys)
    return pd.DataFrame(vstack(a[:]).T,columns=keys)

