import numpy as np
import sympy
from sympy.physics.wigner import wigner_6j
from sympy.utilities.lambdify import lambdify
from scipy.special import erfc
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
from lmfit import Model,Parameters
import os
import pickle
import time
from numba import njit, jit, prange
import spectrumHandler as sh
from spectrumHandler import amu2eV, electronRestEnergy

def kNuc(iNuc, jElec, fTot): return(fTot*(fTot+1)-iNuc*(iNuc+1)-jElec*(jElec+1))

# @njit
def racahCoefficients(iNuc, jElec1, fTot1, jElec2, fTot2):
  iNuc=float(iNuc);jElec1=float(jElec1); fTot1=float(fTot1); jElec2=float(jElec2); fTot2=float(fTot2)
  return((2*fTot1+1)*(2*fTot2+1)/(2*iNuc+1)*wigner_6j(jElec2,fTot2,iNuc,fTot1,jElec1,1)**2)

def energySplitting(A, B, iNuc, jElec, fTot):
  #returns 
  eSplit=0
  if iNuc>=1/2 and jElec>=1/2:
    eSplit += (A/2)*kNuc(iNuc, jElec, fTot)
  if iNuc>=1 and jElec>=1:
    k=kNuc(iNuc, jElec, fTot)
    numerator = 3*k*(k+1)-4*iNuc*(iNuc+1)*jElec*(jElec+1)
    denominator = 8*iNuc*(2*iNuc-1)*jElec*(2*jElec-1)
    eSplit += B*numerator/denominator
  #TODO: add in C dependence some day
  return(eSplit)

def hfsLinesAndStrengths(iNuc,jElec1,jElec2,A1,A2,B1=0,B2=0):
  linesList=[]; strengthsList=[]
  f1List=np.arange(abs(iNuc-jElec1),iNuc+jElec1+1,1)
  f2List=np.arange(abs(iNuc-jElec2),iNuc+jElec2+1,1)
  for fTot1 in f1List:
    for fTot2 in f2List:
      if abs(fTot1-fTot2)<=1: #viable transition!
        shift1=energySplitting(A1, B1, iNuc, jElec1, fTot1)
        shift2=energySplitting(A2, B2, iNuc, jElec2, fTot2)
        linePos=shift2-shift1
        tStrength=racahCoefficients(iNuc, jElec1, fTot1, jElec2, fTot2)
        linesList+=[linePos];strengthsList+=[tStrength]
  return(linesList, strengthsList)

def hyperFinePrediction(x,centroid,gamma,sigma,amplitude,alpha,spShift,spProp,bg=0,Alower=0,Aupper=0,
    Blower=0,Bupper=0):
  global linesFunc, transitionStrengths
  linePositions = np.array(linesFunc(Alower,Aupper,Blower,Bupper)).astype(float)
  relativeHeights= np.array(transitionStrengths).astype(float)
  f = bg
  #sigma=gamma
  for i in range(len(linePositions)):
    x0=float(linePositions[i]+centroid)
    gauss=(1/(sigma*np.sqrt(2*np.pi)))*np.exp(-(x-x0)**2/(2*sigma**2))
    lorentz=(gamma/np.pi)*1/(gamma**2+(x-x0)**2);
    gaussSP=spProp*(1/(sigma*np.sqrt(2*np.pi)))*np.exp(-(x-x0-spShift)**2/(2*sigma**2))
    lorentzSP=spProp*(gamma/np.pi)*1/(gamma**2+(x-x0-spShift)**2)
    mainCont = alpha*gauss+(1-alpha)*lorentz
    sideCont = alpha*gaussSP+(1-alpha)*lorentzSP
    f+=amplitude*relativeHeights[i]*(mainCont+sideCont)
  return(f)

@njit#(parallel=True,fastmath=True,nogil=True)
def lineShape_jit(x, peakList, sp_fractions, amplitude, gammaList, sigmaList, alpha):
  peakCont = np.zeros(len(x), dtype=np.float64)
  for i in prange(len(peakList)):
    diff = x - peakList[i]
    gaussMat = np.exp(-diff**2 / (2.0 * sigmaList[i]**2)) / (sigmaList[i] * np.sqrt(2.0 * np.pi))
    lorentzMat = (gammaList[i]/np.pi) / (gammaList[i]**2 + diff**2)
    peakCont += sp_fractions[i] * ((1.0-alpha) * gaussMat + alpha * lorentzMat)
  return (amplitude * peakCont)

@njit
def get_lineShapeLists(x0,gamma,sigma,spShift,spProp, equal_fwhm=False):
  peakList = np.array([x0, x0 + spShift], dtype=np.float64)
  sp_fractions = np.array([1.0, spProp], dtype=np.float64)
  gammaList = np.full_like(peakList, gamma, dtype=np.float64)
  sigmaList = np.full_like(peakList, sigma, dtype=np.float64)
  if equal_fwhm: gammaList = sigmaList * np.sqrt(2.0 * np.log(2.0))
  return(peakList, sp_fractions, gammaList, sigmaList)

#@njit For some reason, jitting this actually makes it slower? Even removing initial compilation time from timing test.
def lineShape_pseudovoigt(x,x0,amplitude,gamma,sigma,alpha,spShift,spProp,mass=27, equal_fwhm=False,
    laserFrequency=0, frequencyOffset=0, colinearity=True, cec_sim_energies=[], fraction_list=[], spScaling=1):
  if len(cec_sim_energies)==0:
    peakList, sp_fractions, gammaList, sigmaList = get_lineShapeLists(x0,gamma,sigma,spShift,spProp, equal_fwhm=equal_fwhm)
  else:
    sp_scaling_list=spScaling*np.ones(len(fraction_list),dtype=np.float64); sp_scaling_list[0]=1.0
    sp_shifts, sp_fractions, broadeningList = generateSidePeaks(mass, laserFrequency, x0, originalFractionList, cec_sim_energies,
                                                           frequencyOffset=frequencyOffset, colinearity=colinearity)
    peakList = x0+sp_shifts #; print(sp_shifts); print(sp_fractions); quit()
    sp_fractions*=sp_scaling_list
    gammaList=np.full_like(peakList, gamma, dtype=np.float64)
    sigmaList=np.full_like(peakList, sigma, dtype=np.float64)
    sigmaList=np.sqrt(sigmaList**2+broadeningList**2)
  if equal_fwhm: gammaList=sigmaList*np.sqrt(2.0*np.log(2.0))
  f = lineShape_jit(x,peakList,sp_fractions, amplitude, gammaList, sigmaList, alpha)
  return(f)

def hyperFinePredictionFreeAmps_pseudoVoigt(x,centroid,amplitude,gamma,sigma,alpha,spShift,spProp,Alower=0,Aupper=0,
    Blower=0,Bupper=0,h1=1,h2=1,h3=1,h4=1,h5=1,h6=1,h7=1,h8=1,h9=1,h10=1,h11=1,h12=1,iNuc=5/2,mass=27,
    laserFrequency=0, frequencyOffset=0, colinearity=True, cec_sim_data=[], equal_fwhm=False, spScaling=1):
  linePositions, transitionStrengths =hfsLinesAndStrengths(iNuc,1/2,1/2,Alower,Aupper,B1=Blower,B2=Bupper)
  linePositions=np.array(linePositions)[np.argsort(transitionStrengths)[::-1]]
  relativeHeights= np.array([h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11,h12]).astype(float)
  if len(cec_sim_data)==0:cec_sim_energies=[]; fraction_list=[]
  else:cec_sim_energies, fraction_list = cecSimPreProcess(cec_sim_data)
  f = 0
  for i in range(len(linePositions)):
    x0=float(linePositions[i]+centroid)
    f += lineShape_pseudovoigt(x,x0,relativeHeights[i],gamma,sigma,alpha,spShift,spProp,mass=mass, equal_fwhm=equal_fwhm,
          laserFrequency=laserFrequency, frequencyOffset=frequencyOffset, colinearity=colinearity, cec_sim_energies=cec_sim_energies, fraction_list=fraction_list, spScaling=spScaling)
  return(amplitude*f)

def hyperFinePredictionFreeAmps_voigt(x,centroid,amplitude,gamma,sigma,spShift,spProp,Alower=0,Aupper=0,
    Blower=0,Bupper=0,h1=1,h2=1,h3=1,h4=1,h5=1,h6=1,h7=1,h8=1,h9=1,h10=1,h11=1,h12=1,iNuc=5/2,mass=27,
    laserFreq=0, freqOffset=0, colinearity=True, cec_sim_data=[]):
  linePositions, transitionStrengths=hfsLinesAndStrengths(iNuc,1/2,1/2,Alower,Aupper,B1=Blower,B2=Bupper)
  linePositions = [y for _, y in sorted(zip(transitionStrengths, linePositions), key=lambda pair: pair[0])][::-1]
  relativeHeights= np.array([h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11,h12]).astype(float)
  if len(cec_sim_data)==0: cec_sim_energies=[]; sp_fractions=[1, spProp]
  else:
    cec_sim_energies = cec_sim_data[:,2]; sp_fractions=cec_sim_data[:,1]
  f = 0
  for i in range(len(linePositions)):
    x0=float(linePositions[i]+centroid)
    if len(cec_sim_data)==0: xList=np.array([x0, x0+spShift])
    else: xList = generateSidePeaks(mass, laserFreq, x0, cec_sim_energies, freqOffset=freqOffset, colinearity=colinearity)
    voigtMat=voigt(x,xList,gamma,sigma);
    #voigtMat=(gamma/np.pi)*1/(gamma**2+np.subtract.outer(x,xList)**2);
    # allDevs=np.subtract.outer(x,xList)
    # maskedDeviations=allDevs[voigtMat>0]
    # print(np.sort(maskedDeviations.flatten()))
    # plt.plot(np.sort(allDevs.flatten()),'.')
    # plt.plot(np.sort(maskedDeviations.flatten()),'.')
    # plt.show()
    # quit()
    peakCont=np.sum(sp_fractions*voigtMat, axis=1);
    f+=amplitude*relativeHeights[i]*peakCont
  #if np.any(np.isnan(f)):
  #print(centroid,amplitude,gamma,sigma,spShift,spProp,Alower,Aupper,h1,h2,h3,h4)
  #print(f)
  #plt.plot(x, f);plt.show()
  return(f)

def voigt(x,centroidList, gamma,sigma):
  dMat=np.subtract.outer(x,centroidList)
  z=(dMat+1j*gamma)/(np.sqrt(2)*sigma+0j)
  w=np.exp(-z**2)*erfc(-1j*z)
  fMat=1/(np.sqrt(2*np.pi)*sigma) * np.real(w)
  #print(f)
  mask=np.isnan(fMat) + np.isinf(fMat)
  fMat[mask]=.00000001#(gamma/np.pi)*1/(gamma**2+x[mask]**2) #todo: see if there's better way to handle this
  return(fMat)

def backgroundFunction(x, bg=0, slope=0):
  return(bg+slope*x)

def makeDictionaryFromFitStatistics(result):
  d={}
  d['nfev']=result.nfev
  d['nvarys']=result.nvarys
  d['ndata']=result.ndata
  d['nfree']=result.nfree
  d['errorbars']=result.errorbars
  d['residual']=result.residual
  d['chisqr']=result.chisqr
  d['redchi']=result.redchi
  return(d)

def fitData(xData, yData, yUncertainty, mass, iNucList, jGround, jExcited, peakModel='pseudoVoigt', transitionLabel='bruhLabelThis', colinearity=True, laserFrequency=0,
  frequencyOffset=1129900000, centroidGuess=0, fixed_spShift=False, fixed_spProp=False, cec_sim_data_path=False, fixed_Alower=False,
  fixed_Aupper=False, fixed_Aratio=False, equal_fwhm=False,  weightsList=[2.5,1], fixed_Sigma=False, fixed_Gamma=False, spScaleable=False):

  spShiftGuess= fixed_spShift if fixed_spShift else 120
  spPropGuess= fixed_spProp if fixed_spProp else 0.45
  bgGuess=np.quantile(yData,0.1)
  slopeGuess=(yData[-1]-yData[0])/(xData[-1]-xData[0])
  sigmaInit=40 #TODO: automate?
  gammaInit=40
  totAmpGuess=sigmaInit*np.sqrt(2*np.pi)*(np.max(yData)-bgGuess)/len(iNucList)
  if cec_sim_data_path!=False:
    cec_sim_data=np.loadtxt(cec_sim_data_path, skiprows=1,delimiter=',')
    peak1Fraction=cec_sim_data[0,1]; print('test: peak1Fraction=',peak1Fraction)
    totAmpGuess*=1/peak1Fraction
  else: cec_sim_data=[]
  #print("centroidGuess=",centroidGuess)
  if centroidGuess==0:
    mask=yData>bgGuess
    centroidGuess = np.sum(xData[mask]*(yData[mask]-bgGuess)/np.sum(yData[mask]-bgGuess)) #if a non-zero value is supplied, the function will use that as an initial guess
    centroidGuess+=2*(colinearity-0.5)*spShiftGuess*0.5 #2*(colinearity-0.5) returns 1 if colinear, and -1 if anti. sidepeaks will weigh down naive centroid estimate if on left(colinear), and weigh up if on right (anti) of "true" peaks
  else: centroidGuess-=frequencyOffset; #print("bahhh")#this way I can supply an actual centroid and not have to worry about freq offset outside of the function call (Watch this screw me up eventually...)
  
  muDictionary={
  22:[2],
  23:[3.89],
  24:[3, 2.9], #this is just a guess. For the isomer: 2.99
  25:[3.6447],
  27:[3.64070]}

  myMod=Model(backgroundFunction)
  params=Parameters()
  params.add('bg',    value=bgGuess, min=0)
  params.add('slope', value=slopeGuess, vary=True)
  params.add('spScaling', value=1, vary = spScaleable and (cec_sim_data_path!=False))

  for k,iNuc in enumerate(iNucList):
    params.add('iso'+str(k)+'_'+'spScaling', expr = 'spScaling')
    if round(mass)==27:
      if transitionLabel=='P12-S12':
        AlowerGuess=500 ; BlowerGuess=0
        AupperGuess=130; BupperGuess=0
      elif transitionLabel=='P32-S12':
        AlowerGuess=100 ; BlowerGuess=20
        AupperGuess=130; BupperGuess=0
      elif transitionLabel=='P12-D32':
        AlowerGuess=500 ; BlowerGuess=0
        AupperGuess=-108; BupperGuess=13
      elif transitionLabel=='P32-D32':
        AlowerGuess=100 ; BlowerGuess=20
        AupperGuess=-100; BupperGuess=0
      elif transitionLabel=='P32-D52':
        AlowerGuess=94 ; BlowerGuess=23
        AupperGuess=203; BupperGuess=0
      else:print(f'error: {transitionLabel} not recognized')
    else:
      scalingRatio=(2.5/iNuc)*(muDictionary[round(mass)][k]/muDictionary[27][0])
      #print('test: k=%d, I=%.1f mu= %.3f'%(k, iNuc, muDictionary[round(mass)][k]))
      AlowerGuess=500*scalingRatio ; BlowerGuess=0
      AupperGuess=130*scalingRatio; BupperGuess=0

    if len(weightsList)==len(iNucList):ampGuess=totAmpGuess*weightsList[k]/np.sum(weightsList)
    else:ampGuess=totAmpGuess/len(iNucList)
    A1,A2,B1,B2=sympy.symbols('A1 A2 B1 B2')
    linePositions, transitionStrengths=hfsLinesAndStrengths(iNuc,jGround,jExcited,A1,A2,B1=B1,B2=B2)
    #print('test: og line positions = ', linePositions)
    #print('test: og transitionStrengths = ', transitionStrengths)
    linePositions = [x for _, x in sorted(zip(transitionStrengths, linePositions), key=lambda pair: pair[0])][::-1]
    transitionStrengths=np.sort(transitionStrengths)[::-1]
    #print('test: new line positions = ', linePositions)
    #print('test: new transitionStrengths = ', transitionStrengths)
    # linesFunc=lambdify((A1,A2,B1,B2), linePositions, modules='numpy')
    #print("linePositions:",[linePositions[i].subs([(A1,AlowerGuess),(A2,AupperGuess),(B1,BlowerGuess),(B2,BupperGuess)])for i in range(len(linePositions))])
    if peakModel=='pseudoVoigt': toAdd = Model(hyperFinePredictionFreeAmps_pseudoVoigt, prefix='iso'+str(k)+'_', 
                                               independent_vars=['x','frequencyOffset','laserFrequency','colinearity', 'cec_sim_data', 'equal_fwhm'])
    else: toAdd = Model(hyperFinePredictionFreeAmps_voigt, prefix='iso'+str(k)+'_', 
                        independent_vars=['x','frequencyOffset','laserFrequency','colinearity','cec_sim_data','equal_fwhm'])
    params.add('iso'+str(k)+'_'+'iNuc', value=iNuc, vary=False)
    params.add('iso'+str(k)+'_'+'mass', value=mass, vary=False)
  
    transitionStrengths=np.array(transitionStrengths).astype(float); #print('k=%d; transitionStrengths:'%k,transitionStrengths)
    transitionStrengths*=1/np.max(transitionStrengths); #print(transitionStrengths)
    spFactor = -1 if colinearity else 1
    if cec_sim_data_path==False:
      params.add('iso'+str(k)+'_'+'spProp', value=spPropGuess, vary=(fixed_spProp==False), min=0, max=1)
      params.add('iso'+str(k)+'_'+'spShift', value=spFactor*abs(spShiftGuess), max=max(0, spFactor*200), min=min(0, spFactor*200), vary=(fixed_spShift==False))
    else:
      params.add('iso'+str(k)+'_'+'spProp', value=-1, vary=False)
      params.add('iso'+str(k)+'_'+'spShift', value=-1, vary=False)

    params.add('iso'+str(k)+'_'+'Aupper', value=AupperGuess, vary=not(fixed_Aupper))
    if fixed_Aratio == False: params.add('iso'+str(k)+'_'+'Alower', value=AlowerGuess, vary=not(fixed_Alower))
    else:                     params.add('iso'+str(k)+'_'+'Alower', expr=str(fixed_Aratio)+'*iso'+str(k)+'_'+'Aupper')
    params.add('iso'+str(k)+'_'+'Blower',value=BlowerGuess,vary=(jGround>=1 and iNuc>=1))
    params.add('iso'+str(k)+'_'+'Bupper',value=BupperGuess, vary=(jExcited>=1 and iNuc>=1))
    if k==0:  params.add('iso'+str(k)+'_'+'centroid', value=centroidGuess, vary=True)
    elif k==1:params.add('iso'+str(k)+'_'+'centroid', value=centroidGuess+20, vary=True)
    params.add('iso'+str(k)+'_'+'amplitude', value=1, vary=False);
    if k==0:
      if fixed_Sigma!=False:
        params.add('iso'+str(k)+'_'+'sigma', value=fixed_Sigma, vary=False)
      else:
        params.add('iso'+str(k)+'_'+'sigma', value=sigmaInit, vary=True, min=10)
      if equal_fwhm:
        fwhm_scalingFactor=np.sqrt(2*np.log(2))
        params.add('iso'+str(k)+'_'+'gamma', expr='iso'+str(k)+'_'+'sigma*'+str(fwhm_scalingFactor))
      elif fixed_Gamma:
        params.add('iso'+str(k)+'_'+'gamma', value=fixed_Gamma, vary=False)
      else:
        params.add('iso'+str(k)+'_'+'gamma', value=gammaInit, vary=True, min=10)
      if peakModel=='pseudoVoigt': params.add('iso'+str(k)+'_'+'alpha', value=0.5,vary=True, min=0, max=1)
    else:
      params.add('iso'+str(k)+'_'+'sigma', expr='iso0_sigma')
      params.add('iso'+str(k)+'_'+'gamma', expr='iso0_gamma')
      if peakModel=='pseudoVoigt': params.add('iso'+str(k)+'_'+'alpha', expr='iso0_alpha')
      params.add('iso'+str(k)+'_'+'spProp', expr='iso0_spProp')
      spFactor = -1 if colinearity else 1
      params.add('iso'+str(k)+'_'+'spShift', expr='iso0_spShift')

    nominalPositions=[linePositions[i].subs([(A1,AlowerGuess),(A2,AupperGuess),(B1,BlowerGuess),(B2,BupperGuess)])for i in range(len(linePositions))] #locations of lines if initial guess is reasonable
    if k==0: nominalPositions_ground=nominalPositions
    params.add('iso'+str(k)+'_'+'h1', value=ampGuess, vary = True);
    expectedIntensityOrder=np.argsort(transitionStrengths)[::-1]; biggestIndex=expectedIntensityOrder[0]
    for i in range(len(linePositions)): #going through all peaks, but I'm adding them in the order that I expect to be from largest to smallest
      nomPos=nominalPositions[i]
      varyThisPeak=True
      for j in range(i):
        if abs(nomPos-nominalPositions[j])<(sigmaInit+gammaInit)*1.2:#if nominal position of peak i is separated from nom pos of an earlier peak by less than 100% the linewidth, then the amplitudes of the two will be constrained 
          print('peaks %d and %d are expected to overlap significantly and will be bound to eachother'%((i+1),(j+1)))
          varyThisPeak=False
          params.add('iso'+str(k)+'_'+'h'+str(i+1), expr=str(ampGuess*transitionStrengths[i]/transitionStrengths[j])+'*h'+str(j+1))
      if transitionStrengths[i]<0.01:
        print('test, peak %d, has relative strength of %.3f, and should be held fixed'%((i+1), transitionStrengths[i]))
        params.add('iso'+str(k)+'_'+'h'+str(i+1), expr=str(transitionStrengths[i]/transitionStrengths[biggestIndex])+'*iso'+str(k)+'_'+'h'+str(biggestIndex+1), vary = False, min=0); varyThisPeak=False
      if k!=0:
        for j in range(len(nominalPositions_ground)):
            if abs(nomPos-nominalPositions_ground[j])<(sigmaInit+gammaInit)*.75:
              print(f'iso{k}_peak {i+1} and iso0_{j+1} are expected to overlap significantly and will be bound to eachother')
              params.add('diff'+str(i+1), value=ampGuess*transitionStrengths[i]/2, vary = True, min=0, max=ampGuess);#TODO: come back to this #value=1
              params.add('iso'+str(k)+'_'+'h'+str(i+1), expr='iso'+str(0)+'_'+'h'+str(j+1)+'- diff'+str(i+1));
              varyThisPeak=False
      if varyThisPeak: 
        params.add('iso'+str(k)+'_'+'h'+str(i+1), value=ampGuess*transitionStrengths[i], vary = True, min=0);
    for i in range(len(linePositions),12): params.add('iso'+str(k)+'_'+'h'+str(i+1), value=0, vary = False); #print('adding empty peak number %d'%(j+1))
    myMod= myMod + toAdd
  # print('yData', yData, '\n xData:',xData)
  t0=time.perf_counter()
  for i in range(1):
    result=myMod.fit(yData, params, x=xData, method='leastsq', weights=1/yUncertainty,#, fit_kws={'xtol': 1E-6, 'ftol':1E-6},
                     equal_fwhm=equal_fwhm, cec_sim_data=cec_sim_data, frequencyOffset=frequencyOffset, laserFrequency=laserFrequency, colinearity=colinearity)
  t1=time.perf_counter()
  print(f"time elapsed in .fit call:{t1-t0}")
  # interpolator = [lambda i: lambda xInt: comp.eval(params=result.params, x=xInt) for comp in myMod.components]
  bgParamVals={'bg':result.params['bg'].value, 'slope':result.params['slope'].value}
  interpolator=[lambda xInt: backgroundFunction(x=xInt, **bgParamVals)]
  for comp in result.components[1:]:
    evalParms={}
    pref=comp.prefix
    # print(pref)
    for param_name, value in result.best_values.items():
        if param_name.startswith(pref):
            # print(param_name)
            evalParms[param_name[len(pref):]] = value
    # print(evalParms)
    interpolator+=[lambda xInt, parms=evalParms: hyperFinePredictionFreeAmps_pseudoVoigt(x=xInt,
                                                                        equal_fwhm=equal_fwhm, cec_sim_data=cec_sim_data, frequencyOffset=frequencyOffset, laserFrequency=laserFrequency, colinearity=colinearity,
                                                                        **parms)]
  return(result, interpolator)

def bootStrapFunction(bootStrapDictionary, *args, **kwargs):
  print(bootStrapDictionary)
  print(*args)
  print(*kwargs)
  sampleSize=100
  directory='./BootStrappingIntermediateResults/parmPlots/'
  for key in bootStrapDictionary.keys():
    mean=bootStrapDictionary[key][0]
    spread=bootStrapDictionary[key][1]
    kwargsCopy=kwargs.copy()
    samples=list(np.random.normal(loc=mean,scale=spread,size=sampleSize))
    iso0_centroids=[]
    iso1_centroids=[]
    redchis=[]
    for i in range(sampleSize):
      kwargsCopy[key]=samples[i]
      result=fitData(*args, **kwargsCopy)
      iso0_centroids+=[result.params["iso0_centroid"].value]
      iso1_centroids+=[result.params["iso1_centroid"].value]
      redchis+=[result.redchi]
    iso0hLines=[];iso1hLines=[]
    for val in [mean+spread*i for i in [-1,0,1] ]:
      kwargsCopy[key]=val; result=fitData(*args, **kwargsCopy)
      iso0hLines+=[result.params["iso0_centroid"].value]
      iso1hLines+=[result.params["iso1_centroid"].value]
      iso0_centroids+=[result.params["iso0_centroid"].value]
      iso1_centroids+=[result.params["iso1_centroid"].value]
      redchis+=[result.redchi]
      samples+=[val]

    # title=f'mass24_iso0_centroidVs{key}';plt.plot(samples, iso0_centroids,'.',label='iso0'); plt.xlabel(key);plt.ylabel('centroid');plt.title(title); plt.savefig(f'{directory}{title}.png');plt.close()
    # title=f'mass24_iso1_centroidVs{key}';plt.plot(samples, iso1_centroids,'.',label='iso1'); plt.xlabel(key);plt.ylabel('centroid');plt.title(title); plt.savefig(f'{directory}{title}.png');plt.close()
    bootStrapPlots(samples, iso0_centroids, 'iso0_centroid', key, mean=mean, spread=spread, hLines=iso0hLines)
    bootStrapPlots(samples, iso1_centroids, 'iso1_centroid', key, mean=mean, spread=spread, hLines=iso1hLines)
    title=f'mass24_iso1_redchiVs{key}'  ;plt.plot(samples, redchis,'.');plt.xlabel(key);plt.ylabel('redchi')  ;plt.title(title); plt.savefig(f'{directory}{title}.png');plt.close()
  # plt.plot(result.best_fit);plt.show()
  return(redchis)

def bootStrapPlots(samples,y,targetParm,key,mean=0,spread=0,hLines=[]):
  directory='./BootStrappingIntermediateResults/parmPlots/'
  title=f'mass24_{targetParm}Vs{key}'; plt.plot(samples, y,'.');
  plt.gca().axvline(mean,color='k',linestyle='-'); 
  [plt.gca().axvline(mean+spread*x,linestyle='--',color='grey') for x in [-1,1] ]
  if hLines:
    # plt.gca().axhline(hLines[1],color='k',linestyle='-');
    [plt.gca().axhline(hLines[i],linestyle='--',color='grey') for i in [0,2] ]

  plt.xlabel(key);plt.ylabel(targetParm);plt.title(title); plt.savefig(f'{directory}{title}.png')
  plt.close()

def propogateBeamEnergyCorrectionToCentroid(mass, centroid, laserFreq, ΔEkin):
  #print('dummy check: mass = %.1f ; centroid = %.2f'%(mass, centroid))
  neutralRestEnergy=mass*amu2eV
  ionRestEnergy=neutralRestEnergy-electronRestEnergy
  #print(centroid, laserFreq)
  #dcf_0=centroid;
  beta_0 = (centroid**2-laserFreq**2)/(centroid**2+laserFreq**2); gamma_0=1/np.sqrt(1-beta_0**2)
  voltage_0 = ionRestEnergy*(gamma_0-1)
  voltage_1 = voltage_0 + ΔEkin
  beta_1 = np.sqrt(1-((ionRestEnergy)/(voltage_1+ionRestEnergy))**2)
  dcf_1 = np.sqrt(1+beta_1)/np.sqrt(1-beta_1)*laserFreq
  #print('test: voltage_0 = ',voltage_0)

  seriesExpansionScaling = centroid/((ionRestEnergy + voltage_0)*beta_0)
  #print('nu = %.5f+%.5f*ΔEkin'%(centroid, seriesExpansionScaling) )
  #print('total centroid shift = ', dcf_1-centroid)
  return(dcf_1)

def freqToVoltage(mass, laserFreq, peakFreq, freqOffset=0):
  #this function takes a resonance frequency in MHz and converts it to eV
  neutralRestEnergy=mass*amu2eV
  ionRestEnergy=neutralRestEnergy-electronRestEnergy
  peakFreq+=freqOffset
  beta_0_amp = (peakFreq**2-laserFreq**2)/(peakFreq**2+laserFreq**2); #anti/collinearity doesn't matter, since I just square this to get gamma_0 anyway 
  gamma_0=1/np.sqrt(1-beta_0_amp**2); #print(gamma_0)
  voltage_0 = ionRestEnergy*(gamma_0-1); #print(voltage_0)
  return(voltage_0)

def freqShiftToVoltageShift(mass, laserFreq, peakFreq, peakShift, freqOffset=0):
  #this function takes a sidepeak fit result in MHz and converts it to eV
  voltage_0 = freqToVoltage(mass, laserFreq, peakFreq, freqOffset=freqOffset)
  voltage_1 = freqToVoltage(mass, laserFreq, peakFreq+peakShift, freqOffset=freqOffset)
  Δv = voltage_1-voltage_0
  return(Δv)

def voltageShiftToFrequencyShift(mass, laserFreq, voltage0, voltageShift, freqOffset=0, colinearity=True):
  neutralRestEnergy=mass*amu2eV
  ionRestEnergy=neutralRestEnergy-electronRestEnergy
  beta_0=np.sqrt(1-((ionRestEnergy)/(voltage0+ionRestEnergy))**2)
  beta_1=np.sqrt(1-((ionRestEnergy)/(voltage0+voltageShift+ionRestEnergy))**2)
  if colinearity: beta_0*=-1; beta_1*=-1
  #doppler corrected frequencies
  dcf0 = np.sqrt(1+beta_0)/np.sqrt(1-beta_0)*laserFreq
  dcf1 = np.sqrt(1+beta_1)/np.sqrt(1-beta_1)*laserFreq
  Δf=dcf1-dcf0
  return(Δf)

@njit#(fastmath=True)
def generateSidePeaks(mass, laserFrequency, peakFreq, sp_fractions, cec_sim_list, frequencyOffset=0, colinearity=True):#, cecBinning=False):
  '''
  based on input mass, laser frequency, and resonance frequency, determine resonance in voltage space, then 
  apply e_losses from cec_sim_list, and finally transform back to frequency space to predict sidepeak positions
  '''
  originalPeakFreq=peakFreq#...this seems to change slightly if I don't do this
  mask = sp_fractions != 0 # Filtering non-zero elements
  cec_sim_list = cec_sim_list[mask]; sp_fractions = sp_fractions[mask]
  cec_sim_list[cec_sim_list<0] /=3 #update May 23, 2025: dividing energy loss channels by 3, for some reasons.

  neutralRestEnergy=mass*amu2eV
  ionRestEnergy=neutralRestEnergy-electronRestEnergy

  peakFreq+=frequencyOffset #need to use actual resonance frequency, not fitting frequency (which is offset), or else Doppler transforms are wrong
  beta_0_amp = (peakFreq**2-laserFreq**2)/(peakFreq**2+laserFreq**2); #anti/collinearity doesn't matter, since I just square this to get gamma_0 anyway 
  gamma_0=1/np.sqrt(1-beta_0_amp**2); #print(gamma_0)
  voltage_0 = ionRestEnergy*(gamma_0-1); #print(voltage_0)
  voltageList = voltage_0 - cec_sim_list; ##e-losses actually look like higher voltages, because you had to accelerate more to end up resonant after inelastic collision 
  betaList=np.sqrt(1-((ionRestEnergy)/(voltageList+ionRestEnergy))**2)
  if colinearity:
    dcfList = np.sqrt(1-betaList)/np.sqrt(1+betaList)*laserFreq #doppler corrected frequencies
  else:
    dcfList = np.sqrt(1+betaList)/np.sqrt(1-betaList)*laserFreq #doppler corrected frequencies
  dcfList-=frequencyOffset

  neg_mask = cec_sim_list < 0; pos_mask = cec_sim_list > 0
  sp_shifts=np.zeros(len(dcfList[neg_mask])+1)
  sp_shifts[1:]=dcfList[neg_mask]
  sp_shifts[0] = originalPeakFreq
  
  negativeOffsets=dcfList[pos_mask]-sp_shifts[0]
  negOffsetWeights=sp_fractions[pos_mask]

  #question: should I sum the positive e_losses linearly or in quadrature? TODO: Confirm that it makes sense to transform to MHz
  broadeningFactor = np.sum(negativeOffsets * negOffsetWeights) / np.sum(negOffsetWeights) if len(negOffsetWeights) > 0 else 0.0
  #Note: jit does not like my binning code, and honestly why bin if you can jit anyway? Keeping here in case I ever want to bring it back.
  # if cecBinning!=False: #if true, we are binning the simulated sidepeaks in bins of width cecBinning, and we need to calculate weighted means for sp_positions, and sum the appropriate fractiosn 
  #   bins=np.arange(sp_shifts.min(),sp_shifts.max()+cecBinning, cecBinning)#-cecBinning/2
  #   bindices = np.digitize(sp_shifts,bins)
  #   finalLength=len(np.unique(bindices))
  #   tempShifts=np.zeros(finalLength); tempFracts=np.zeros(finalLength)
  #   for i, bindex in enumerate(np.unique(bindices)):
  #     tempFracts[i] = np.sum(sp_fractions[bindices==bindex])
  #     tempShifts[i] = np.sum(sp_fractions[bindices==bindex]*sp_shifts[bindices==bindex])/tempFracts[i]
  #   sp_shifts=tempShifts
  #   sp_fractions=tempFracts
  broadeningList=np.zeros(len(sp_shifts)); broadeningList[0] = broadeningFactor
  sp_shifts-=sp_shifts[0]# I'm only returning the shifts to the main peak frequency to make the use case slightly more general
  return(sp_shifts, sp_fractions, broadeningList)

def cecSimPreProcess(cec_sim_data):
  cec_sim_energies = cec_sim_data[:,2]; sp_fractions=cec_sim_data[:,1]
  mask=sp_fractions>0 #we only want to consider non-zero fractions (and positive ofc. see:kolmogorov)
  cec_sim_energies=cec_sim_energies[mask]; sp_fractions=sp_fractions[mask]
  return(cec_sim_energies, sp_fractions)


if __name__ == '__main__':
  # x=np.linspace(-1000,1000,200); x0=0; amplitude=1; gamma=60; sigma=60;alpha=0.5;spShift=120;spProp=0.5
  # y1=lineShape_pseudoVoigt(x,x0,amplitude,gamma,sigma,alpha,spShift,spProp)
  # x=np.random.random(100); bins=np.linspace(-1,1,100)
  # res=testFunc(x, bins); print(res)
  # quit()
  x=np.linspace(-2000,2000,200); x0=0; amplitude=1; gamma=50; sigma=50;alpha=0.5;spShift=120;spProp=0.5; equal_fwhm=True
  if equal_fwhm: gamma=sigma*np.sqrt(2*np.log(2))
  mass=26.981538408; laserFreq=3E6*376.052850; x0=0; frequencyOffset=1129900000
  Alower=136; Aupper=500; iNuc=5/2
  colinearity=False; spScaling=1
  cec_sim_data=np.loadtxt('../OnlineAlAnalysis/27Al_CEC_peaks.csv', skiprows=1,delimiter=',')
  cec_sim_energies, originalFractionList = cecSimPreProcess(cec_sim_data)
  sp_scaling_list=spScaling*np.ones_like(originalFractionList); sp_scaling_list[0]=1
  for x0 in np.random.random(10):
    y=lineShape_pseudovoigt(x,x0,amplitude,gamma,sigma,alpha,spShift,spProp,mass=mass, equal_fwhm=equal_fwhm,
                          laserFreq=laserFreq, frequencyOffset=frequencyOffset, colinearity=colinearity, cec_sim_energies=cec_sim_energies, fraction_list=originalFractionList, spScaling=1.)
    y2=lineShape_pseudovoigt(x,x0,amplitude,gamma,sigma,alpha,spShift,spProp, equal_fwhm=equal_fwhm)
  print("starting now")
  dt1=[];dt2=[]
  linePositions, transitionStrengths=hfsLinesAndStrengths(iNuc,1/2,1/2,Alower,Aupper,B1=0,B2=0)
  linePositions = [y for _, y in sorted(zip(transitionStrengths, linePositions), key=lambda pair: pair[0])][::-1]
  hfSplits=linePositions

  for size in [10**n for n in range(1)]:
    print(f'size:{size}')
    x0List=np.random.random(size)
    t0=time.perf_counter()
    for x0 in x0List:    
      y=hyperFinePredictionFreeAmps_pseudoVoigt(x,x0,amplitude,gamma,sigma,alpha,spShift,spProp,Alower=Alower,Aupper=Aupper,
        Blower=0,Bupper=0,h1=1,h2=1,h3=1,h4=1,h5=1,h6=1,h7=1,h8=1,h9=1,h10=1,h11=1,h12=1,iNuc=5/2,mass=27,
        equal_fwhm=False, colinearity=True)
    t1=time.perf_counter()
    y2=0
    for x0 in x0List:
      hfList=x0+hfSplits
      for hfp in hfSplits:
        y2+=lineShape_pseudovoigt(x,hfp,amplitude,gamma,sigma,alpha,spShift,spProp, equal_fwhm=equal_fwhm, colinearity=colinearity)
    t2=time.perf_counter()
    dt1+=[t1-t0]
    dt2+=[t2-t1]
  print(dt1)
  print(dt2)
  print(np.array(dt1)/np.array(dt2))
  plt.plot(x,y,label='old construction'); plt.plot(x,y2,'--',label='new, piecewise/njitted construction');
  # plt.plot(dt1,label='old construction'); plt.plot(dt2,'--',label='new, piecewise/njitted construction');
  # plt.ylabel('runtime(s)');plt.xlabel('log(iterations)')
  plt.legend(); plt.show()

#TODO:
# add bootstrap functionality for arbitrary set of parameters, passed as dictionary with [mean, unc] as val
# extract bootstrapped uncertainties on A_ratio, spShift, and spProp, for 24,22 Al
