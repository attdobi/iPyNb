#Function to initialize ipython notebook with matplotlib, seaborn, pandas, bokeh, scipy...
from __future__ import division #otherwise 1/73 = 0 in python2
from numpy import *
import scipy
import pylab
import matplotlib.pyplot as plt # plotting libraries from matlab
from scipy.stats import multivariate_normal
from dateutil import parser
import matplotlib.dates as md
import scipy.io as sio
from scipy.optimize import curve_fit # for fitting
import pandas as pd

from bokeh.plotting import figure, output_file, show, VBox, ColumnDataSource, output_server, cursession, curdoc, output_notebook
from bokeh.models.widgets import Slider, TextInput, HBox, VBox, Dialog, Button, VBoxForm, MultiSelect, PreText, Select
from bokeh.models import Range1d, HoverTool, BoxSelectTool, Callback, Circle, ColumnDataSource, Slider, CheckboxButtonGroup, Rect
from bokeh.server.utils.plugins import object_page
from bokeh.io import vform, hplot
from bokeh.charts import Histogram, Bar
from IPython.html.widgets import interact

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


sys.path.insert(2, '/global/project/projectdirs/lux/Tools/anaconda/lib/python2.7/site-packages/seaborn-0.6.0-py2.7.egg-info')
import seaborn as sns