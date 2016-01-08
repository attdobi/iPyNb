from __future__ import division
import pylab
import scipy.interpolate as ip
from scipy.optimize import curve_fit
import sys
import os
home_dir=os.popen("echo $HOME").read()
home_dir=home_dir.split('\n')[0] # parse the string and remove the /n at the end....
sys.path.insert(2,home_dir+'/iPyNb/NERSC_Import') #make sure you have iPyNb checked out from AD
sys.path.insert(2,home_dir+'/LZ_git/fastNEST/libNEST') #add to import Lib-NEST
#Need to add this to make Skin Response maps load in for libNEST
os.environ['NESTPATH']='/global/homes/a/adobi/LZ_git/fastNEST/libNEST/'
os.environ['NESTDATA']='/global/homes/a/adobi/LZ_git/fastNEST/LCEAndEFieldMaps/'
import libNEST
sys.path.insert(2, '//global/project/projectdirs/lux/data') #frozen with NEST v98\n", #on pdsf. or point to aLib in LUXCode
from aLib import inrange, eff, rates
from aLib import pyNEST as pn
import matplotlib as mpl
import matplotlib.pyplot as plt
from numpy import *
import scipy.stats as st
import numpy as np
import scipy as sp

'''Define gaussian distribution for fitting'''
def gauss(x, *p):
    A, mu, sigma = p
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))

'''Define color Scatter Plot'''
def scatterColor(x,y,a=0.8,size=8):
    xy=vstack([x,y])
    xy[isnan(xy)]=0
    xy[isinf(xy)]=0
    z= st.gaussian_kde(xy)(xy)
    plt.scatter(x,y,c=z,s=size,edgecolor='',alpha=a)
    return

def NEST_setup(ParticleType='NR',energy=10,f_drift=-1,xe_density=2.888,drift_time=-1,g1=0.075, nFold=3, eff_extract=0.95,e_lifetime=845, spEff_direct=0.9, Det='LZ', custom=False):
    '''input: ParticleType('ER' or 'NR'), energy, drift field V/cm, xe_density, drift_time, g1, nFold_coincidence, electron extraction efficiency,electron lifetime (us),  spEff_direct, Det='LZ' or 'LUX')'''
    #Setup the LUX or LZ detector:
    if (ParticleType=='ER'):
        pt=1
    elif (ParticleType=='NR'):
        pt=0
    else:
        print('Invalid particle type defulting to NR')
        pt=0
        
    #Setup NEST    
    NEST=libNEST.NEST(pt,energy,f_drift,xe_density,drift_time) # 0 is NR, 1 is ER ... Energy, EField(V/cm), density, dT
    
    if (Det=='LZ'):
        myDet=libNEST.Detector()
        myDet.LZSettings() 
    elif (Det=='LUX'):
        myDet=libNEST.Detector()
        myDet.LUXSettings()
    else:
        print('Invalid detector, defulting to LZ')
        myDet=libNEST.Detector()
        myDet.LZSettings()
    
    #Modify custom detector parameters
    if custom==True:
        myDet.g1=g1
        myDet.g1_gas=g1
        myDet.nFold=nFold
        myDet.e_life=e_lifetime
        myDet.ee=eff_extract
        myDet.spEff_direct=spEff_direct;
        NEST.SetDetectorParameters(myDet)
    else:
        NEST.SetDetectorParameters(myDet) 
    
    return NEST

'''Define function dRdE>quanta->signal, for LZ'''
def dN2S1S2(NEST=NEST_setup(), file_path='data/PP_7Be_evt_ton_year_keV_lin_noDiscrim.txt', nSim=1e5, kg_days=5600*1000,Esim_max=200):
    '''input: NEST, file path to diff spectrum, nSim, total rate in evts/kg/day, Esim_max to cut off)'''
    
    #nSim=ceil(total_rate*5600*1000)# use total_rate= Calc_Rate_evts_kg_day to Simulate 1 nominal LZ exposure
    Edatatxt, rate=np.loadtxt(file_path,skiprows=0,unpack=True) #evts/ton/year #Energy scale must be linear in text file for code to work properly!
    if (max(Edatatxt) < Esim_max):
        Esim_max=max(Edatatxt)
    Edata=np.arange(min(Edatatxt),Esim_max,0.001)#0.01 keV binning
    dR = rate/1000/365 #convert from Ton to kg and year to days 
    dR=np.interp(Edata,Edatatxt,dR)
    Rcum = dR.cumsum()/dR.sum()  # Energy cumulative distribution function
    cutRange = Rcum < Rcum[-1]
    Rcum = Rcum[cutRange]
    dR = dR[cutRange]
    Edata = Edata[cutRange]
    r_uniform = np.random.rand(nSim)
    Eee = np.interp(r_uniform, Rcum, Edata)
    
    ## Generate Signal in the detector ##
    Nph=[]
    Ne=[]
    S1=[]
    S2=[]
    S1c=[]
    S2c=[]
    
    for en in Eee:
        NEST.SetEnergy(en)
        NEST.DetectorResponse()
        Nph.append(NEST.GetNumPhotons())
        Ne.append(NEST.GetNumElectrons())
        S1.append(NEST.GetS1())
        S2.append(NEST.GetS2())
        S1c.append(NEST.GetS1c())
        S2c.append(NEST.GetS2c())
    
    Nph=np.array(Nph)
    Ne=np.array(Ne)
    S1=np.array(S1)
    S2=np.array(S2)
    S1c=np.array(S1c)
    S2c=np.array(S2c)

    Eee_min = min(Edata) #for integral
    Eee_max = max(Edata) #for integral
    Eee_vect = np.linspace(Eee_min,Eee_max*1.001,2e6)
    dR_vect = np.interp(Eee_vect[Eee_vect>=0.1],Edata[Edata>=0.1],dR[Edata>=0.1]) # Start integrating at 0.1 keV
    Rate_evts_kg_day = float(dR_vect.sum() * np.diff(Eee_vect[:2])) #evts/kg/day 100% acceptance
    Det_exposure_factor=nSim/(Rate_evts_kg_day*kg_days)
    print('total between {:.2f} and {:.2f} keV = {:g} [evts/kg/day]'.format(Eee_min, Eee_max, Rate_evts_kg_day)) #evts/kg/day
    print('Detector exposure factor = {:g}'.format(Det_exposure_factor)) #evts/kg/day

    plt.figure()
    plt.loglog(Edata,dR*1000*365,'-k')
    plt.hold('on')
    if (NEST.GetParticleType()==1): #if ER
        plt.loglog(Edata,dR*1000*365/200,'--k') #if ER then plot with 1/200 ER rejection
    
    #ylim([1e-3, 1e0])
    #xlim([.5, 1e2])
    plt.xlabel('Recoil Energy [keV]')
    plt.ylabel('Event Rate [/ton/year/keV]')
    #text(1,0.013,'PP+7Be',fontsize=16)
    plt.rcParams.update({'font.size': 18})

    return Nph, Ne, S1, S2, S1c, S2c, Rate_evts_kg_day, Det_exposure_factor

'''Define function WIMP mass to S1&S2 for LZ'''
def WIMP2NphNe(NEST=NEST_setup(), mWmp=50,nSim=1e5, kg_days=5600*1000):
    '''input: NEST=NEST_setup(), file path to diff spectrum, total rate in evts/kg/day'''
        
    #nSim=ceil(total_rate*5600*1000)# use total_rate= Calc_Rate_evts_kg_day to Simulate 1 nominal LZ exposure
    Er = rates.WIMP.genRandEnergies(nSim, mW=mWmp)
    
    ## Generate Signal in the detector ##
    Nph=[]
    Ne=[]
    S1=[]
    S2=[]
    S1c=[]
    S2c=[]
    
    for en in Er:
        NEST.SetEnergy(en)
        NEST.DetectorResponse()
        Nph.append(NEST.GetNumPhotons())
        Ne.append(NEST.GetNumElectrons())
        S1.append(NEST.GetS1())
        S2.append(NEST.GetS2())
        S1c.append(NEST.GetS1c())
        S2c.append(NEST.GetS2c())
    
    Nph=np.array(Nph)
    Ne=np.array(Ne)
    S1=np.array(S1)
    S2=np.array(S2)
    S1c=np.array(S1c)
    S2c=np.array(S2c)

    Er_min = min(Er) #for integral
    Er_max = max(Er) #for integral
    Er_vect = np.linspace(Er_min,Er_max*1.01,2e6)
    dR_vect = rates.WIMP.WIMPdiffRate(Er_vect, mW=mWmp, sig=1e-45)
    WmpRate = float((dR_vect.sum() * np.diff(Er_vect[:2])) / (1e-9)) #evts/kg/day per 1 pb, 100% acceptance
    Det_exposure_factor=nSim/(WmpRate*kg_days)
    print('total rate above {:.2f} keV = {:g} [evts/kg/day per pb]'.format(Er_min, WmpRate)) #evts/kg/day/pb
    print('Detector exposure factor per pb = {:g}'.format(Det_exposure_factor)) #evts/kg/day

    plt.figure()
    plt.loglog(Er_vect,dR_vect*1000*365,'-k')
    plt.hold('on')
    #ylim([1e-3, 1e0])
    #xlim([.5, 1e2])
    plt.xlabel('Recoil Energy [keV]')
    plt.ylabel('Event Rate [/ton/year/keV/1e-9 pb]')
    #text(1,0.013,'PP+7Be',fontsize=16)
    plt.rcParams.update({'font.size': 18})

    return Nph, Ne, S1, S2, S1c, S2c, WmpRate, Det_exposure_factor
    
'''Define function to generate flat ER and NR bands'''    
def genBands(NEST=NEST_setup(),nSim=2e5, maxS1=50, S2raw_min=450, mWmp=50):
    
    #start with NR
    NEST.SetParticleType(0)
    
    maxEr=100 #keVnr, for flat spectrum
    #Default, use a 50 GeV WIMP for discrimnation
    if mWmp==-1:
        Er = maxEr*st.uniform.rvs(size=nSim); #0-100 keVnr
    else:
        Er = rates.WIMP.genRandEnergies(nSim, mW=mWmp)
    
    ## Generate Signal in the detector ##
    Nph=[]
    Ne=[]
    S1=[]
    S2=[]
    S1c=[]
    S2c=[]
    
    for en in Er:
        NEST.SetEnergy(en)
        NEST.DetectorResponse()
        Nph.append(NEST.GetNumPhotons())
        Ne.append(NEST.GetNumElectrons())
        S1.append(NEST.GetS1())
        S2.append(NEST.GetS2())
        S1c.append(NEST.GetS1c())
        S2c.append(NEST.GetS2c())
        
    Nph=np.array(Nph)
    Ne=np.array(Ne)
    S1=np.array(S1)
    S2=np.array(S2)
    S1c=np.array(S1c)
    S2c=np.array(S2c)
    
    S1_bins=linspace(1,maxS1,maxS1)
    S1_bin_cen_n=empty_like(S1_bins)
    mean_S2oS1_n=empty_like(S1_bins)
    std_S2oS1_n=empty_like(S1_bins)
    coeff_n=[]
    var_matrix_n=[]
    #Find the NR S2/S1 band at each S1
    det_cuts= (S1c>0) & (S2c>0) & (S2>=S2raw_min)
    for index, S1s in enumerate(S1_bins):
        cut=det_cuts & inrange(S1c,[S1s-1/2,S1s+1/2])
        S1_bin_cen_n[index]=mean(S1c[cut])
        mean_S2oS1_n[index]=mean(log10(S2c[cut]/S1c[cut]))
        std_S2oS1_n[index]=std(log10(S2c[cut]/S1c[cut]))
        #fit Gauss to distibution
        # p0 is the initial guess for the fitting coefficients (A, mu and sigma above)
        #hist, bin_edges = np.histogram(log10(S2c[cut]/S1c[cut]), 20, density=True)
        #bin_centres = (bin_edges[:-1] + bin_edges[1:])/2
        #p0 = [max(hist), mean_S2oS1_n[index], std_S2oS1_n[index]]
        #coeff, var_matrix = curve_fit(gauss, bin_centres, hist, p0=p0)
        #coeff_n.append(coeff)
        #var_matrix_n.append(var_matrix)
    #convert to numpy array. 3xN matrix
    #coeff_n=np.array(coeff_n) #indexed as: amp, mean, sigma
    #var_matrix_n=np.array(var_matrix_n)
   
    #get NR mean, with a smooth spline(need this for discrimination calculation)
    if nSim>=5e5:
        sNR = ip.UnivariateSpline(S1_bin_cen_n, mean_S2oS1_n,s=.0001) #essentially linear interp
    else:
        sNR = ip.UnivariateSpline(S1_bin_cen_n, mean_S2oS1_n,s=.001) #essentially linear interp
    #Use gaussian fit
    #sNR = ip.UnivariateSpline(S1_bin_cen_n, coeff_n[:,1],s=0.005)
    
    #Calc Nr Efficiency vs E
    E_bins=linspace(1,maxEr,maxEr)
    E_bin_cen_n=empty_like(E_bins)
    Eff_n=empty_like(E_bins)
    det_cuts= (S1c>0) & (S2>=S2raw_min)
    for index, Es in enumerate(E_bins):
        cut=det_cuts & inrange(Er,[Es-0.5,Es+0.5])
        E_bin_cen_n[index]=mean(Er[cut])
        Eff_n[index]=sum(cut)/sum(inrange(Er,[Es-0.5,Es+0.5])) #cut/total_in_bin
   
    
    #Calculate the ER band and discrimination ####################################################
    maxEe=30 #keVee, for flat ER spectrum
    Flat_Ee = maxEe*st.uniform.rvs(size=nSim); #0-50 keVee
    ## Generate Signal in the detector ##
    Nph=[]
    Ne=[]
    S1=[]
    S2=[]
    S1c=[]
    S2c=[]
    
    NEST.SetParticleType(1) #set to ER events
    for en in Flat_Ee:
        NEST.SetEnergy(en)
        NEST.DetectorResponse()
        Nph.append(NEST.GetNumPhotons())
        Ne.append(NEST.GetNumElectrons())
        S1.append(NEST.GetS1())
        S2.append(NEST.GetS2())
        S1c.append(NEST.GetS1c())
        S2c.append(NEST.GetS2c())
        
    Nph=np.array(Nph)
    Ne=np.array(Ne)
    S1=np.array(S1)
    S2=np.array(S2)
    S1c=np.array(S1c)
    S2c=np.array(S2c)
    
    S1_bins=linspace(1,maxS1,maxS1)
    S1_bin_cen_e=empty_like(S1_bins)
    mean_S2oS1_e=empty_like(S1_bins)
    std_S2oS1_e=empty_like(S1_bins)
    num_leak_e=empty_like(S1_bins)
    num_total_e=empty_like(S1_bins)
    leak_gauss_e=empty_like(S1_bins)
    #Find the ER S2/S1 band at each S1
    det_cuts= (S1c>0) & (S2c>0) & (S2>=S2raw_min)
    for index, S1s in enumerate(S1_bins):
        cut=det_cuts & inrange(S1c,[S1s-1/2,S1s+1/2])
        S1_bin_cen_e[index]=mean(S1c[cut])
        mean_S2oS1_e[index]=mean(log10(S2c[cut]/S1c[cut]))
        std_S2oS1_e[index]=std(log10(S2c[cut]/S1c[cut]))
        #calculate discrimination
        num_leak_e[index]=sum(log10(S2c[cut]/S1c[cut])<sNR(S1c[cut])) #num below NR mean
        num_total_e[index]=sum(cut)
        #Gaussian overlap
        nsig=(mean_S2oS1_e[index]-sNR(S1_bin_cen_e[index]))/std_S2oS1_e[index] #(meanER-meanNR)/sigER
        leak_gauss_e[index]=sp.special.erfc(nsig/sqrt(2))/2
           
    
    #Calc Er Efficiency vs E
    E_bins=linspace(1,maxEe,maxEe)
    E_bin_cen_e=empty_like(E_bins)
    Eff_e=empty_like(E_bins)
    det_cuts= (S1c>0) & (S2>=S2raw_min)
    for index, Es in enumerate(E_bins):
        cut=det_cuts & inrange(Flat_Ee,[Es-0.5,Es+0.5])
        E_bin_cen_e[index]=mean(Flat_Ee[cut])
        Eff_e[index]=sum(cut)/sum(inrange(Flat_Ee,[Es-0.5,Es+0.5])) #cut/total_in_bin

    return S1_bin_cen_n, mean_S2oS1_n, std_S2oS1_n, S1_bin_cen_e, mean_S2oS1_e, std_S2oS1_e, E_bin_cen_e, Eff_e, E_bin_cen_n, Eff_n, num_leak_e, num_total_e, leak_gauss_e, sNR
    

def E2NphNe(ParticleType='ER',Energy = linspace(1,100,1000),f_drift=700,g1=0.075,SPE_res= 0.5,eff_extract=0.95,SE_size=50,SE_res=10,e_lifetime=1000, dt0=500):
    #input: ParticleType('ER' or 'NR'), Energy array, drift field V/cm, g1, SPE_res, electron extraction efficiency, SE size, sigSE, electron lifetime, center of detector)
    Nph, Ne = pn.Nph_Ne(ParticleType,f_drift*np.ones_like(Energy),Energy)
    S1 = st.binom.rvs(array(Nph, dtype=int64),g1) #mod g1
    #No S1 PMT resolution, assume spike counting
    #add PMT resolution, assume sigma = 0.5 PE for a single PE collected
    #S1_B8 = st.norm.rvs(S1_B8,sqrt(S1_B8)*SPE_res,size=size(S1_B8)) #single PE resolution with sigma=sqrt(N)*sig
    Ne_ext = st.binom.rvs(array(Ne*exp(-dt0/e_lifetime), dtype=int64), eff_extract) #add in mean electron attenuation
    S2 = st.norm.rvs(Ne_ext*SE_size,sqrt(SE_res**2 * Ne_ext),size=size(Ne_ext)) #adding variance "SE_res**2 * Ne_ext"
    S2 = S2*exp(dt0/e_lifetime) #correct back the electron lifetime
    
    return Nph, Ne, S1, S2