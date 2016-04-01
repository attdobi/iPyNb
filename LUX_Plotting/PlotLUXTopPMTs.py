from __future__ import division
import matplotlib as mpl
import numpy as np
from numpy import *
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

#%pylab inline
pylab.rcParams['figure.figsize'] = (8.0, 8.0) # set size of figures"
plt.rcParams.update({'font.size': 20})

def LUXPlotTopPMTs():
    '''
    This function will plot the top PMT array, with numbers, on the current
    figure.

     LUXPlotTopPMTs

     No inputs or outputs.

    Versioning
      20120418 CHF - Created using Dave's code used in LUXHitPattern
      20150331 AD - Python version usgin matplotlib
    '''
    #pmt_pos_cm= np.loadtxt('/Users/attiladobi/LUXCode/Scratch/AttilaDobi/for_svn/iPyNb/LUX_Plotting/pmt_pos_cm.txt')
    pmt_pos_cm= np.loadtxt('pmt_pos_cm.txt')

    pmt_width = 5.7 # cm
    tt = np.arange(0,2*pi,pi/64)
    xx = pmt_width/2*cos(tt)
    yy = pmt_width/2*sin(tt)

    ## Plot
    #plt.figure()
    for ii_pmt in range(60)+[121]:
        ii_pmt_name = ii_pmt
        plt.plot(pmt_pos_cm[ii_pmt,0]+xx,\
            pmt_pos_cm[ii_pmt,1]+yy,color=np.array([1,1,1])*0.7) 
        plt.hold('on')   
        plt.text(pmt_pos_cm[ii_pmt,0]+0.3,pmt_pos_cm[ii_pmt,1],\
                 str(ii_pmt_name),color=np.array([1,1,1])*0.9,\
                 horizontalalignment='center', verticalalignment='center');

    ww = 18.632 * 2.54 # ptfe panel face-face width, in->cm
    ll = ww/2
    alpha_d = 30; # degrees
    beta_d = alpha_d/2
    xy = np.array([[-ll*np.tan(np.radians(beta_d)), ll*np.tan(np.radians(beta_d))],[ll, ll]])
    xy = np.matmul(np.array([[np.cos(np.radians(15)), -np.sin(np.radians(15))],\
                             [np.sin(np.radians(15)),np.cos(np.radians(15))]]),xy)
    for ii in range(12):
        hh = plt.plot(xy[0,:],xy[1,:],'-k',color=np.array([1,1,1])*0.7)
        #set(hh,'color',np.array([1,1,1])*0.7)
        Ralpha = np.array([[np.cos(np.radians(alpha_d)), -np.sin(np.radians(alpha_d))]\
                           ,[ np.sin(np.radians(alpha_d)), np.cos(np.radians(alpha_d))]])
        xy_new_1 = np.matmul(Ralpha,xy[:,0])
        xy_new_2 = np.matmul(Ralpha,xy[:,1])
        xy = np.vstack((xy_new_1,xy_new_2)).T

    # Pretty it up 

    #plt.axis([-30,30,-27,27])
    plt.axis('equal')
    plt.xlim([-30,30])
    plt.ylim([-27,27])
    plt.xlabel('x (cm)')
    plt.ylabel('y (cm)')