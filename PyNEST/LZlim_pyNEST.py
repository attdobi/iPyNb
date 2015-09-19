from __future__ import division
import pylab
import scipy.interpolate as ip
import sys
sys.path.insert(2, '//global/project/projectdirs/lux/data') #frozen with NEST v98\n",
from aLib import pyNEST as pn
from aLib import inrange, eff, rates
import matplotlib as mpl
import matplotlib.pyplot as plt
from numpy import *
import scipy.stats as st
import numpy as np


'''Define color Scatter Plot'''
def scatterColor(x,y,a=0.8):
    xy=vstack([x,y])
    xy[isnan(xy)]=0
    xy[isinf(xy)]=0
    z= st.gaussian_kde(xy)(xy)
    plt.scatter(x,y,c=z,s=8,edgecolor='',alpha=a)
    return

'''Define function dRdE>quanta->signal, for LZ'''
def dN2NphNe(ParticleType='ER',file_path='data/PP_7Be_evt_ton_year_keV_lin_noDiscrim.txt', nSim=1e5, kg_days=5600*1000, f_drift=700,g1=0.075,SPE_res= 0.5, minSpikePE=0.25, eff_extract=0.95,SE_size=50,SE_res=sqrt(50),e_lifetime=1000, dt0=500):
    '''input: ParticleType('ER' or 'NR'), file path to diff spectrum, total rate in evts/kg/day, drift field V/cm, g1, SPE_res, electron extraction efficiency, SE size, sigSE, electron lifetime, center of detector)'''
    #nSim=ceil(total_rate*5600*1000)# use total_rate= Calc_Rate_evts_kg_day to Simulate 1 nominal LZ exposure
    Edatatxt, rate=np.loadtxt(file_path,skiprows=0,unpack=True) #evts/ton/year #Energy scale must be linear in text file for code to work properly!
    Edata=np.arange(min(Edatatxt),max(Edatatxt),0.001)#0.01 keV binning
    dR = rate/1000/365 #convert from Ton to kg and year to days 
    dR=np.interp(Edata,Edatatxt,dR)
    Rcum = dR.cumsum()/dR.sum()  # Energy cumulative distribution function
    cutRange = Rcum < Rcum[-1]
    Rcum = Rcum[cutRange]
    dR = dR[cutRange]
    Edata = Edata[cutRange]
    Eee=Edata
    r_uniform = np.random.rand(nSim)
    Eee = np.interp(r_uniform, Rcum, Eee)
    Nph, Ne = pn.Nph_Ne(ParticleType,f_drift*np.ones_like(Eee),Eee)
    S1_spike = st.binom.rvs(array(Nph, dtype=int64),g1) #binomial photon collection
    NS1_coin=st.binom.rvs(S1_spike,1-math.erfc(((1-minSpikePE)/SPE_res)/sqrt(2))/2) #takes at least minSpikePE to produce a spike, use binomial probability calculated from erfc
    S1 = st.norm.rvs(S1_spike,sqrt(S1_spike)*SPE_res,size=size(S1_spike)) #single PE resolution with sigma=sqrt(N)*sig_PE
    Ne_ext = st.binom.rvs(array(Ne*exp(-dt0/e_lifetime), dtype=int64), eff_extract)
    S2 = st.norm.rvs(Ne_ext*SE_size,sqrt(SE_res**2 * Ne_ext),size=size(Ne_ext))
    S2 = S2*exp(dt0/e_lifetime)/eff_extract

    Eee_min = min(Edata) #for integral
    Eee_max = max(Edata) #for integral
    Eee_vect = np.linspace(Eee_min,Eee_max*1.01,2e6)
    dR_vect = np.interp(Eee_vect[Eee_vect>=0.1],Edata[Edata>=0.1],dR[Edata>=0.1]) # Start integrating at 0.1 keV
    Rate_evts_kg_day = float(dR_vect.sum() * np.diff(Eee_vect[:2])) #evts/kg/day 100% acceptance
    LZ_exposure_factor=nSim/(Rate_evts_kg_day*5600*1000)
    print('total rate above {:.2f} keV = {:g} [evts/kg/day]'.format(Eee_min, Rate_evts_kg_day)) #evts/kg/day
    print('LZ exposure factor = {:.1f}'.format(LZ_exposure_factor)) #evts/kg/day

    plt.figure()
    plt.loglog(Edata,dR*1000*365,'-k')
    plt.hold('on')
    if (ParticleType=='ER'):
        plt.loglog(Edata,dR*1000*365/200,'--k') #if ER then plot with 1/200 ER rejection
    
    #ylim([1e-3, 1e0])
    #xlim([.5, 1e2])
    plt.xlabel('Recoil Energy [keV]')
    plt.ylabel('Event Rate [/ton/year/keV]')
    #text(1,0.013,'PP+7Be',fontsize=16)
    plt.rcParams.update({'font.size': 18})

    return Nph, Ne, Ne_ext, S1_spike, S1, S2, NS1_coin, Rate_evts_kg_day, LZ_exposure_factor

'''Define function WIMP mass to S1&S2 for LZ'''
def WIMP2NphNe(mWmp=50,nSim=1e5, kg_days=5600*1000, f_drift=700,g1=0.075,SPE_res= 0.5, minSpikePE=0.25, eff_extract=0.95,SE_size=50,SE_res=sqrt(50),e_lifetime=1000, dt0=500):
    '''input: ParticleType('ER' or 'NR'), file path to diff spectrum, total rate in evts/kg/day, drift field V/cm, g1, SPE_res, electron extraction efficiency, SE size, sigSE, electron lifetime, center of detector)'''
    #nSim=ceil(total_rate*5600*1000)# use total_rate= Calc_Rate_evts_kg_day to Simulate 1 nominal LZ exposure
    Er = rates.WIMP.genRandEnergies(nSim, mW=mWmp)
    Nph, Ne = pn.Nph_Ne('NR',f_drift*np.ones_like(Er),Er)
    S1_spike = st.binom.rvs(array(Nph, dtype=int64),g1) #binomial photon collection
    NS1_coin=st.binom.rvs(S1_spike,1-math.erfc(((1-minSpikePE)/SPE_res)/sqrt(2))/2) #takes at least minSpikePE to produce a spike, use binomial probability calculated from erfc
    S1 = st.norm.rvs(S1_spike,sqrt(S1_spike)*SPE_res,size=size(S1_spike)) #single PE resolution with sigma=sqrt(N)*sig_PE
    Ne_ext = st.binom.rvs(array(Ne*exp(-dt0/e_lifetime), dtype=int64), eff_extract)
    S2 = st.norm.rvs(Ne_ext*SE_size,sqrt(SE_res**2 * Ne_ext),size=size(Ne_ext))
    S2 = S2*exp(dt0/e_lifetime)/eff_extract

    Er_min = min(Er) #for integral
    Er_max = max(Er) #for integral
    Er_vect = np.linspace(Er_min,Er_max*1.01,2e6)
    dR_vect = rates.WIMP.WIMPdiffRate(Er_vect, mW=mWmp, sig=1e-45)
    WmpRate = float((dR_vect.sum() * np.diff(Er_vect[:2])) / (1e-9)) #evts/kg/day per 1 pb, 100% acceptance
    LZ_exposure_factor=nSim/(WmpRate*5600*1000)
    print('total rate above {:.2f} keV = {:g} [evts/kg/day per pb]'.format(Er_min, WmpRate)) #evts/kg/day/pb
    print('LZ exposure factor per pb = {:.g}'.format(LZ_exposure_factor)) #evts/kg/day

    plt.figure()
    plt.loglog(Er_vect,dR_vect*1000*365,'-k')
    plt.hold('on')
    #ylim([1e-3, 1e0])
    #xlim([.5, 1e2])
    plt.xlabel('Recoil Energy [keV]')
    plt.ylabel('Event Rate [/ton/year/keV/1e-9 pb]')
    #text(1,0.013,'PP+7Be',fontsize=16)
    plt.rcParams.update({'font.size': 18})

    return Nph, Ne, Ne_ext, S1_spike, S1, S2, NS1_coin, WmpRate, LZ_exposure_factor
    
'''Define function to generate flat ER and NR bands'''    
def genBands(nSim=1e5,maxS1=50,f_drift=700,g1=0.075,SPE_res= 0.5,eff_extract=0.95,SE_size=50,SE_res=sqrt(50),e_lifetime=1000, dt0=500,minSpikePE=0.25,min_NS1_coin=3, min_Ne_ext=5 ):
    #Calculate the NR band, and count below that for acceptance ########
    maxEr=100 #keVnr, for flat spectrum... DD
    Flat_Er = maxEr*st.uniform.rvs(size=nSim); #0-100 keVnr
    Nph, Ne= pn.Nph_Ne('NR',f_drift*np.ones_like(Flat_Er),Flat_Er)
    S1_spike = st.binom.rvs(array(Nph, dtype=int64),g1) #binomial photon collection
    NS1_coin=st.binom.rvs(S1_spike,1-math.erfc(((1-minSpikePE)/SPE_res)/sqrt(2))/2) #takes at least minSpikePE to produce a spike, use binomial probability calculated from erfc
    S1 = st.norm.rvs(S1_spike,sqrt(S1_spike)*SPE_res,size=size(S1_spike)) #single PE resolution with sigma=sqrt(N)*sig_PE
    Ne_ext = st.binom.rvs(array(Ne*exp(-dt0/e_lifetime), dtype=int64), eff_extract)
    S2 = st.norm.rvs(Ne_ext*SE_size,sqrt(SE_res**2 * Ne_ext),size=size(Ne_ext))
    S2 = S2*exp(dt0/e_lifetime)/eff_extract
    S1_bins=linspace(1,maxS1,maxS1)
    S1_bin_cen_n=empty_like(S1_bins)
    mean_S2oS1_n=empty_like(S1_bins)
    std_S2oS1_n=empty_like(S1_bins)
    #Find the NR S2/S1 band at each S1
    det_cuts= (NS1_coin>=min_NS1_coin) & (Ne_ext>=min_Ne_ext)
    for index, S1s in enumerate(S1_bins):
        cut=det_cuts & inrange(S1,[S1s-0.5,S1s+0.5])
        S1_bin_cen_n[index]=mean(S1[cut])
        mean_S2oS1_n[index]=mean(log10(S2[cut]/S1[cut]))
        std_S2oS1_n[index]=std(log10(S2[cut]/S1[cut]))
        
    #Calc Nr Efficiency vs E
    E_bins=linspace(1,maxEr,maxEr)
    E_bin_cen_n=empty_like(E_bins)
    Eff_n=empty_like(E_bins)
    det_cuts= (NS1_coin>=min_NS1_coin) & (Ne_ext>=min_Ne_ext)
    for index, Es in enumerate(E_bins):
        cut=det_cuts & inrange(Flat_Er,[Es-0.5,Es+0.5])
        E_bin_cen_n[index]=mean(Flat_Er[cut])
        Eff_n[index]=sum(cut)/sum(inrange(Flat_Er,[Es-0.5,Es+0.5])) #cut/total_in_bin
        
    #Calculate the ER band #################################
    maxEe=100 #keVee, for flat ER spectrum
    Flat_Ee = maxEe*st.uniform.rvs(size=nSim); #0-100 keVee
    Nph, Ne= pn.Nph_Ne('ER',f_drift*np.ones_like(Flat_Ee),Flat_Ee)
    S1_spike = st.binom.rvs(array(Nph, dtype=int64),g1) #binomial photon collection
    NS1_coin=st.binom.rvs(S1_spike,1-math.erfc(((1-minSpikePE)/SPE_res)/sqrt(2))/2) #takes at least minSpikePE to produce a spike, use binomial probability calculated from erfc
    S1 = st.norm.rvs(S1_spike,sqrt(S1_spike)*SPE_res,size=size(S1_spike)) #single PE resolution with sigma=sqrt(N)*sig_PE
    Ne_ext = st.binom.rvs(array(Ne*exp(-dt0/e_lifetime), dtype=int64), eff_extract)
    S2 = st.norm.rvs(Ne_ext*SE_size,sqrt(SE_res**2 * Ne_ext),size=size(Ne_ext))
    S2 = S2*exp(dt0/e_lifetime)/eff_extract
    S1_bins=linspace(1,maxS1,maxS1)
    S1_bins=linspace(1,maxS1,maxS1)
    S1_bin_cen_e=empty_like(S1_bins)
    mean_S2oS1_e=empty_like(S1_bins)
    stdev_S2oS1_e=empty_like(S1_bins)
    #Find the ER S2/S1 band at each S1
    det_cuts= (NS1_coin>=min_NS1_coin) & (Ne_ext>=min_Ne_ext)
    for index, S1s in enumerate(S1_bins):
        cut=det_cuts & inrange(S1,[S1s-0.5,S1s+0.5])
        S1_bin_cen_e[index]=mean(S1[cut])
        mean_S2oS1_e[index]=mean(log10(S2[cut]/S1[cut]))
        stdev_S2oS1_e[index]=std(log10(S2[cut]/S1[cut]))
        
    #Calc Nr Efficiency vs E
    E_bins=linspace(1,maxEe,maxEe)
    E_bin_cen_e=empty_like(E_bins)
    Eff_e=empty_like(E_bins)
    det_cuts= (NS1_coin>=min_NS1_coin) & (Ne_ext>=min_Ne_ext)
    for index, Es in enumerate(E_bins):
        cut=det_cuts & inrange(Flat_Ee,[Es-0.5,Es+0.5])
        E_bin_cen_e[index]=mean(Flat_Ee[cut])
        Eff_e[index]=sum(cut)/sum(inrange(Flat_Ee,[Es-0.5,Es+0.5])) #cut/total_in_bin

    return S1_bin_cen_n, mean_S2oS1_n, std_S2oS1_n, S1_bin_cen_e, mean_S2oS1_e, stdev_S2oS1_e, E_bin_cen_e, Eff_e, E_bin_cen_n, Eff_n
    

def E2NphNe(ParticleType='ER',Energy = linspace(1,100,1000),f_drift=700,g1=0.075,SPE_res= 0.5,eff_extract=0.95,SE_size=50,SE_res=sqrt(50),e_lifetime=1000, dt0=500):
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