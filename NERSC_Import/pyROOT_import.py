# Import ROOT, rootnotes_nersc for display and the root_numpy lib
from __future__ import division #otherwise 1/73 = 0 in python2
import sys
#sys.path.insert(2,'/global/project/projectdirs/lux/Tools/root_pdsf/root/lib') #don't need with rootnb
#sys.path.insert(2,'/global/common/pdsf_sl6/usg/software/ROOT/5.34.10/lib')
sys.path.insert(2,'/global/project/projectdirs/lux/Tools/anaconda/lib/python2.7/site-packages')
from ROOT import TCanvas, TPad, TFile, TPaveText, TChain, TCut, TF1, TH1F, TLine,TLegend,TH2F, TText,TLatex, TTree
from ROOT import gBenchmark, gStyle, gROOT, gSystem, TVirtualFitter, TEllipse
from root_numpy import root2array, root2rec, tree2rec, list_structures
import root_numpy
import scipy.io as sio
from numpy import *
import rootnotes
