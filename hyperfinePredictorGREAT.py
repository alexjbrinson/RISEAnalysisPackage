import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc
import sympy
from sympy.physics.wigner import wigner_6j
from sympy.utilities.lambdify import lambdify
import os
from lmfit import Model,Parameters, Parameter, model
import pickle
import csv
import pandas as pd
import polars as pl
import time

amu2eV = np.float64(931494102.42)
electronRestEnergy=510998.950 #in eV #TODO: put this back in
def kNuc(iNuc, jElec, fTot): return(fTot*(fTot+1)-iNuc*(iNuc+1)-jElec*(jElec+1))
def racahCoefficients(iNuc, jElec1, fTot1, jElec2, fTot2):
  iNuc=float(iNuc);jElec1=float(jElec1); fTot1=float(fTot1); jElec2=float(jElec2); fTot2=float(fTot2)
  # print(jElec2,fTot2,iNuc,fTot1,jElec1,1)
  # print([type(x) for x in [jElec2,fTot2,iNuc,fTot1,jElec1,1]])
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

def gimmeLinesAndStrengths(iNuc,jElec1,jElec2,A1,A2,B1=0,B2=0):
  linesList=[]
  strengthsList=[]
  f1List=np.arange(abs(iNuc-jElec1),iNuc+jElec1+1,1)
  f2List=np.arange(abs(iNuc-jElec2),iNuc+jElec2+1,1)
  for fTot1 in f1List:
    for fTot2 in f2List:
      #print(fTot1, fTot2)
      if abs(fTot1-fTot2)<=1:
        #print('viable transition!')
        shift1=energySplitting(A1, B1, iNuc, jElec1, fTot1)
        shift2=energySplitting(A2, B2, iNuc, jElec2, fTot2)
        linePos=shift2-shift1
        tStrength=racahCoefficients(iNuc, jElec1, fTot1, jElec2, fTot2)
        #print('position=',linePos,'tStrength=',tStrength)
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

def hyperFinePredictionFreeAmps_pseudoVoigt(x,centroid,amplitude,gamma,sigma,alpha,spShift,spProp,Alower=0,Aupper=0,
    Blower=0,Bupper=0,h1=1,h2=1,h3=1,h4=1,h5=1,h6=1,h7=1,h8=1,h9=1,h10=1,h11=1,h12=1,iNuc=5/2,mass=27,
    laserFreq=0, freqOffset=0, colinearity=True, cec_sim_data=[], equal_fwhm=False, spScaling=1, cecBinning=False):
  linePositions, transitionStrengths=gimmeLinesAndStrengths(iNuc,1/2,1/2,Alower,Aupper,B1=Blower,B2=Bupper)
  linePositions = [y for _, y in sorted(zip(transitionStrengths, linePositions), key=lambda pair: pair[0])][::-1]
  relativeHeights= np.array([h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11,h12]).astype(float)
  if len(cec_sim_data)==0: cec_sim_energies=[]; sp_fractions=[1, spProp]
  else:
    cec_sim_energies = cec_sim_data[:,2]; sp_fractions=cec_sim_data[:,1]
    cec_sim_energies=cec_sim_energies[sp_fractions>0]; sp_fractions=sp_fractions[sp_fractions>0]; originalFractionList=sp_fractions
    sp_scaling_list=spScaling*np.ones_like(sp_fractions); sp_scaling_list[0]=1
    # print('sp_scaling_list:',sp_scaling_list)
    
    # print('spScaling:',spScaling)
    # print('sp_fractions:',sp_fractions)
  f = 0
  for i in range(len(linePositions)):
    x0=float(linePositions[i]+centroid)
    if len(cec_sim_data)==0: xList=np.array([x0, x0+spShift]); gammaList=gamma*np.ones_like(xList); sigmaList=sigma*np.ones_like(xList)
    else:
      sp_shifts, sp_fractions, broadeningList = generateSidePeakFreqs(mass, laserFreq, x0, originalFractionList, cec_sim_energies,
                                                             freqOffset=freqOffset, colinearity=colinearity, cecBinning=cecBinning)
      sp_fractions=sp_fractions*sp_scaling_list
      xList = x0+sp_shifts #; print(sp_shifts); print(sp_fractions); quit()
      gammaList=gamma*np.ones_like(xList); sigmaList=sigma*np.ones_like(xList)
      sigmaList=np.sqrt(sigmaList**2+broadeningList**2)
      if equal_fwhm: gammaList=sigmaList*np.sqrt(2*np.log(2))

    gaussMat=(1/(sigmaList*np.sqrt(2*np.pi)))*np.exp(-np.subtract.outer(x,xList)**2/(2*sigmaList**2))
    lorentzMat=(gammaList/np.pi)*1/(gammaList**2+np.subtract.outer(x,xList)**2);
    peakCont= np.sum(sp_fractions*((1-alpha)*gaussMat+alpha*lorentzMat),axis=1)
    if len(peakCont)!=len(x): print('pooopy'); break
    f+=amplitude*relativeHeights[i]*(peakCont)
  return(f)

def hyperFinePredictionFreeAmps_voigt(x,centroid,amplitude,gamma,sigma,spShift,spProp,Alower=0,Aupper=0,
    Blower=0,Bupper=0,h1=1,h2=1,h3=1,h4=1,h5=1,h6=1,h7=1,h8=1,h9=1,h10=1,h11=1,h12=1,iNuc=5/2,mass=27,
    laserFreq=0, freqOffset=0, colinearity=True, cec_sim_data=[]):
  linePositions, transitionStrengths=gimmeLinesAndStrengths(iNuc,1/2,1/2,Alower,Aupper,B1=Blower,B2=Bupper)
  linePositions = [y for _, y in sorted(zip(transitionStrengths, linePositions), key=lambda pair: pair[0])][::-1]
  relativeHeights= np.array([h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11,h12]).astype(float)
  if len(cec_sim_data)==0: cec_sim_energies=[]; sp_fractions=[1, spProp]
  else:
    cec_sim_energies = cec_sim_data[:,2]; sp_fractions=cec_sim_data[:,1]
  f = 0
  for i in range(len(linePositions)):
    x0=float(linePositions[i]+centroid)
    if len(cec_sim_data)==0: xList=np.array([x0, x0+spShift])
    else: xList = generateSidePeakFreqs(mass, laserFreq, x0, cec_sim_energies, freqOffset=freqOffset, colinearity=colinearity)
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
  # if np.any(np.isinf(w)): 
  #   print('promblem parms:', dMat[np.isinf(w)], gamma,sigma)
  #   print('promblem result:',w[np.isinf(w)]); quit()
  fMat=1/(np.sqrt(2*np.pi)*sigma) * np.real(w)
  #print(f)
  mask=np.isnan(fMat) + np.isinf(fMat)
  fMat[mask]=.00000001#(gamma/np.pi)*1/(gamma**2+x[mask]**2) #todo: see if there's better way to handle this
  #f=(gamma/np.pi)*1/(gamma**2+x**2)
  #print('gamma: ',gamma, 'sigma: ',sigma)
  # for i in range(len(centroidList)):
  #   plt.plot(x,fMat[:,i])
  # plt.show()
  #print(f); quit()
  #if len(f[np.isnan(f)])>0: print('whyyyy\n',d[np.isnan(f)]); quit()
  #fMat=(gamma/np.pi)*1/(gamma**2+np.subtract.outer(x,centroidList)**2)

  return(fMat)

def backgroundFunction(x, bg=0, slope=0):
  return(bg+slope*x)

def importDataFrame(path, scan, energyCorrection=False, timeOffset=0): #windowToF=[]
  filename=path+'/scan%d/scan%d_DataFrame.csv'%(scan,scan)
  polarsdFrame = pl.read_csv(filename)
  scanTimes=np.array(polarsdFrame['scanTime']) - timeOffset
  if energyCorrection:
    if type(energyCorrection)==model.ModelResult:       
      voltageCorrections = energyCorrection.eval(x=scanTimes); print('test:voltageCorrections:\n',voltageCorrections)
    else: voltageCorrections = energyCorrection*np.ones_like(scanTimes)
  else: voltageCorrections = np.zeros_like(scanTimes)
  try: polarsdFrame=polarsdFrame.with_columns(voltageCorrections=voltageCorrections)
  except Exception as e: print(e.dir()); quit()
  return(polarsdFrame)

def cutToF_polars(df, minT, maxT, cuttingColumn='time_step'):
  print('cuttingColumn: ', cuttingColumn)
  df=df.filter(pl.col(cuttingColumn) > minT).filter(pl.col(cuttingColumn) < maxT)
  return(df)

def cutToF(dataFrame, minT, maxT, cuttingColumn='time_step'):
  print('cuttingColumn: ', cuttingColumn)
  dataFrame2=dataFrame.copy()
  dataFrame2[cuttingColumn]=dataFrame2[cuttingColumn].map(lambda v: v if (v>minT and v<maxT) else float('NaN'))
  dataFrame2.dropna(inplace=True)
  return(dataFrame2)

def makeSpectrum(dataFrame, laserFreq, mass, colinear=True, **kwargs):
  #TODO
  neutralRestEnergy=mass*amu2eV
  ionRestEnergy=neutralRestEnergy-electronRestEnergy
  dataFrame['totalVoltage']=dataFrame['totalVoltage']+dataFrame['voltageCorrections']
  dataFrame['partialWeights']=dataFrame['totalVoltage']*dataFrame['PMT0']
  tSteps=np.max(dataFrame['time_step'])-np.min(dataFrame['time_step'])+1 #this represents the number of "toCount" entries I would get from a single measurement at a given vStep
  # vSteps=np.max(dataFrame['vstep']) #use this to decide how many bins to make
  # vCuts = pd.cut(dataFrame.loc[:,'scan_volt_set'], bins=vSteps, retbins=False) #bin on 'scan_volt_set' in case of different scan starting points
  vSteps=np.linspace(np.min(dataFrame['totalVoltage']-0.5),np.max(dataFrame['totalVoltage']+0.5), dataFrame['vstep'].nunique()+1)
  vCuts = pd.cut(dataFrame.loc[:,'totalVoltage'], bins=vSteps, retbins=False) #bin on 'scan_volt_set' in case of different scan starting points
  groupFrame=dataFrame.groupby(vCuts, observed=False) #I don't think this observed keyword is important, but it shuts up a FutureWarning
  
  voltageData=np.array(groupFrame.mean()['totalVoltage']).astype("float64");
  #this conversion to numpy's float64 type is necessary because I'm now using "pyarrow_extension_array=True" when converting from polars to pandas, and pyarrow arrays seem to be immutable
  noZerosMask=np.array(groupFrame.sum()['PMT0'])!=0
  voltageData[noZerosMask]=np.array(groupFrame.sum()['partialWeights'])[noZerosMask]/np.array(groupFrame.sum()['PMT0'])[noZerosMask] #count-weighted mean of voltages probed in each vstep grouping

  #np.savetxt("SanityChecking/v3.csv", voltageData, delimiter=","); print('here'); quit()
  betaData=np.sqrt(1-((ionRestEnergy)/(voltageData+ionRestEnergy))**2)#np.sqrt(2*np.array(tFrame.mean()['totalVoltage'])/(mass*amu2eV)) #Important to treat this relativistically smh
  
  if colinear:
    dcf = np.sqrt(1-betaData)/np.sqrt(1+betaData)*laserFreq #doppler corrected frequencies
  else:
    dcf = np.sqrt(1+betaData)/np.sqrt(1-betaData)*laserFreq #doppler corrected frequencies
  
  yData=np.array(groupFrame.sum()['PMT0']); yUncertainty=np.sqrt(yData); yUncertainty[yUncertainty<1]=1
  
  normalizer=float(tSteps)/np.array(groupFrame.sum()['toCount']); #print("toCount:\n",np.array(groupFrame.sum()['toCount']));
  #print("normalizer:\n", normalizer); print('voltage:\n', groupFrame.mean()['totalVoltage']); print(len(normalizer)); quit()
  yData=yData*normalizer; yUncertainty=yUncertainty*normalizer

  d={'beam_energy':voltageData,'dcf':dcf,'countrate':yData,'uncertainty':yUncertainty, 'totalPasses':1/normalizer}
  spectrumFrame=pd.DataFrame(data=d)
  spectrumFrame.dropna(inplace=True)

  avgScanTime=np.mean(dataFrame['scanTime'])
  spectrumFrame['avgScanTime'] = avgScanTime

  return(spectrumFrame)

def tofSpectrum(dataFrame):
  # Creates ToF spectrum from raw ascii dataframe
  tSteps=int(np.max(dataFrame['time_step'])-np.min(dataFrame['time_step'])+1)
  #print('test: tsteps = ',tSteps)
  tBins = pd.cut(dataFrame.loc[:,'time_step'], bins=tSteps, retbins=False)
  groupFrame=dataFrame.groupby(tBins, observed=False) #I don't think this observed keyword is important, but it shuts up a FutureWarning
  tData=np.array(groupFrame.mean()['time_step'])
  yData=np.array(groupFrame.sum()['PMT0'])
  d={'tof':tData,'counts':yData}
  spectrumFrame=pd.DataFrame(data=d)
  return(spectrumFrame)

def tofSpectrum_polars(dataFrame):
  groupFrame=dataFrame.group_by('time_step')
  # tData=np.array(dataFrame.group_by('time_step').agg(pl.col('ToF').mean()))
  yData=np.array(groupFrame.agg(pl.col('PMT0').sum()))
  order=np.argsort(yData[:,0])
  d={'tof':yData[order,0],'counts':yData[order,1]}
  spectrumFrame=pd.DataFrame(data=d)
  return(spectrumFrame)

def make2DSpectrum(dataFrame, windowToF=[]):
  vSteps=np.max(dataFrame['vstep'])
  if len(windowToF)==2:
    minToF=windowToF[0]; maxToF=windowToF[1]
    dataFrame['ToF']=dataFrame['ToF'].map(lambda v: v if (v>minToF and v<maxToF) else float('NaN'))
    dataFrame.dropna(inplace=True)
  fBins = pd.cut(dataFrame.loc[:,"dcf"], bins=vSteps)#, retbins=True)
  aggDat = dataFrame.groupby(fBins).agg({'dcf':['mean'], 'PMT0':['sum']}).reset_index()
  
  fBins = pd.cut(dataFrame.loc[:,"dcf"], bins=vSteps)#, retbins=True)
  tBins=pd.cut(dataFrame.loc[:,"ToF"], bins=1024)#, retbins=True)
  aggDat2 = dataFrame.groupby([fBins,tBins]).agg({'dcf':['mean'],'ToF':['mean'], 'PMT0':['sum']}).reset_index()
  #testImgX=np.array(aggDat2.loc[:,('dcf','mean')]).reshape((vSteps,1024))
  #testImgY=np.array(aggDat2.loc[:,('ToF','mean')]).reshape((vSteps,1024))
  testImg=np.array(aggDat2.loc[:,('PMT0','sum')]).reshape((vSteps,1024))
  #plt.imshow(testImg,aspect='equal')
  return(aggDat)

def exportSpectrumFrame(scanDirec, runs, laserFreq, mass, targetDirectoryName, colinearity=True, windowToF=[],
  energyCorrection=False, timeOffset=0, directoryPrefix='spectralData', keepLessIntegratedBins=True,**kwargs):
  t0=time.perf_counter()
  polarsFrame=pl.DataFrame()
  for i,run in enumerate(runs):
    print('loading run %d'%run)
    if energyCorrection:
      if type(energyCorrection)==model.ModelResult or type(energyCorrection)==float:
        currentDataFrame=importDataFrame(scanDirec, run, energyCorrection=energyCorrection, timeOffset=timeOffset)
      else:
        currentDataFrame=importDataFrame(scanDirec, run, energyCorrection=energyCorrection[i], timeOffset=timeOffset)
    else:
      currentDataFrame=importDataFrame(scanDirec, run)
    polarsFrame=pl.concat([polarsFrame,currentDataFrame])
  t1=time.perf_counter()
  print(f'Δt1={t1-t0}')
  spectrumPath = './'+directoryPrefix+'/mass'+str(round(mass))+'/'+targetDirectoryName+'/' #this is where I'll save all outputs of this function
  if energyCorrection: 
    spectrumPath += 'energyCorrected/' #this is where I'll save all outputs of this function if an energy correction is provided
    if not os.path.exists(spectrumPath): os.makedirs(spectrumPath)
    if type(energyCorrection)==model.ModelResult:
      scanTimes=np.array(polarsFrame['scanTime']) - timeOffset      
      voltageCorrections = energyCorrection.eval(x=scanTimes); print('test:voltageCorrections:\n',voltageCorrections)
      print('Hot dog, using calibration function for energy correction!')
      # np.savetxt(spectrumPath+'voltageCorrections.csv', np.c_[dFrame['scanTime'][::1024],voltageCorrections[::1024]], delimiter=',') #this is huge lmao
      with open(spectrumPath+'averageEnergyCorrection.txt','w') as file:
        file.write('<Delta E(t)>='+str(np.mean(voltageCorrections))+'V\tRange: '+str([np.min(voltageCorrections), np.max(voltageCorrections)])); file.close()
      print('test:scan times:\n',scanTimes)

    else: voltageCorrections = energyCorrection #presumably this is a float or int potentially
    # try: polarsFrame=polarsFrame.with_columns(voltageCorrections=voltageCorrections)
    # except Exception as e: print(e.dir()); quit()
  # else: polarsFrame=polarsFrame.with_columns(voltageCorrections=np.zeros_like(polarsFrame['totalVoltage']))

  if not os.path.exists(spectrumPath):
    os.makedirs(spectrumPath)
  #plot ToF spectrum, and make cuts if input
  fig=plt.figure()
  t2=time.perf_counter()
  timeSpec=tofSpectrum_polars(polarsFrame)
  t3=time.perf_counter()
  plt.plot(timeSpec['tof'],timeSpec['counts'])
  if len(windowToF)==2:
    #print('making tof cut: ['+str(windowToF[0])+', '+str(windowToF[1])+']')
    # dFrame=cutToF(dFrame, windowToF[0], windowToF[1],**kwargs); timeSpec=tofSpectrum(dFrame)
    polarsFrame=cutToF_polars(polarsFrame, windowToF[0], windowToF[1],**kwargs); timeSpec=tofSpectrum_polars(polarsFrame)
    plt.plot(timeSpec['tof'],timeSpec['counts'])
  plt.title(str(round(mass))+'Al ToF Histogram - '+targetDirectoryName)
  fig.set_size_inches(18.5, 10.5)
  fig.savefig(spectrumPath+'tof_plot.png',dpi=200)
  fig.clf()
  plt.close(fig)
  
  fig=plt.figure()
  #make spectrum, write it to a file, and then plot it
  t4=time.perf_counter()
  dFrame=polarsFrame.to_pandas(use_pyarrow_extension_array = True)
  spectrumFrame = makeSpectrum(dFrame, laserFreq, mass, colinear=colinearity)
  t5=time.perf_counter()
  spectrumFrame.to_csv(spectrumPath+'spectralData_total.csv') if energyCorrection==False else spectrumFrame.to_csv(spectrumPath+'spectralData_total_energyCorrected.csv')
  plt.errorbar(spectrumFrame['dcf'],spectrumFrame['countrate'],yerr=spectrumFrame['uncertainty'],fmt='r.',ecolor='black',capsize=1,markersize=8)
  if keepLessIntegratedBins: pass
  else:  spectrumFrame = spectrumFrame[spectrumFrame['totalPasses']==np.max(spectrumFrame['totalPasses'])]
  spectrumFrame.to_csv(spectrumPath+'spectralData.csv') if energyCorrection==False else spectrumFrame.to_csv(spectrumPath+'spectralData_energyCorrected.csv')
  plt.errorbar(spectrumFrame['dcf'],spectrumFrame['countrate'],yerr=spectrumFrame['uncertainty'],fmt='b.',ecolor='black',capsize=1,markersize=8)

  plt.ylabel('countrate');plt.xlabel('Frequency (MHz)')
  plt.title(str(round(mass))+'Al spectral data - '+targetDirectoryName)
  fig.set_size_inches(18.5, 10.5)
  fig.savefig(spectrumPath+'spectrum_plot.png',dpi=200)
  fig.clf()
  plt.close(fig)
  print(f'Δt1={t1-t0};Δt2={t3-t2};Δt3={t5-t4}')

def loadSpectrumFrame(mass, targetDirectoryName, energyCorrection=False, directoryPrefix='spectralData'):
  spectrumPath = './'+directoryPrefix+'/mass'+str(round(mass))+'/'+targetDirectoryName+'/' #this is where I'll save all outputs of this function
  if energyCorrection:
    spectrumPath += '/energyCorrected/' #this is where I'll save all outputs of this function if an energy correction is provided
    spectrumFrame = pd.read_csv(spectrumPath+'spectralData_energyCorrected.csv')
  else: spectrumFrame = pd.read_csv(spectrumPath+'spectralData.csv')
  return(spectrumFrame)

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


def fitData(xData, yData, yUncertainty, mass, iNucList, jGround, jExcited, peakModel='pseudoVoigt', transitionLabel='bruhLabelThis', colinearity=True, laserFreq=0,
  freqOffset=1129900000, energyCorrection=False, centroidGuess=0, spShiftGuess=120, cec_sim_data_path=False, fixed_spShift=False, fixed_Alower=False,
  fixed_Aupper=False, subPath='', fixed_Aratio=False, equal_fwhm=False,  weightsList=[2.5,1], fixed_Sigma=False, fixed_Gamma=False, spScaleable=False, cecBinning=False):
  print('spScaleable:', spScaleable)
  print('cec_sim_data_path:', cec_sim_data_path)
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
  else: centroidGuess-=freqOffset; #print("bahhh")#this way I can supply an actual centroid and not have to worry about freq offset outside of the function call (Watch this screw me up eventually...)
  
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
    else:
      scalingRatio=(2.5/iNuc)*(muDictionary[round(mass)][k]/muDictionary[27][0])
      #print('test: k=%d, I=%.1f mu= %.3f'%(k, iNuc, muDictionary[round(mass)][k]))
      AlowerGuess=500*scalingRatio ; BlowerGuess=0
      AupperGuess=130*scalingRatio; BupperGuess=0

    if len(weightsList)==len(iNucList):ampGuess=totAmpGuess*weightsList[k]/np.sum(weightsList)
    else:ampGuess=totAmpGuess/len(iNucList)
    A1,A2,B1,B2=sympy.symbols('A1 A2 B1 B2')
    linePositions, transitionStrengths=gimmeLinesAndStrengths(iNuc,jGround,jExcited,A1,A2,B1=B1,B2=B2)
    #print('test: og line positions = ', linePositions)
    #print('test: og transitionStrengths = ', transitionStrengths)
    linePositions = [x for _, x in sorted(zip(transitionStrengths, linePositions), key=lambda pair: pair[0])][::-1]
    transitionStrengths=np.sort(transitionStrengths)[::-1]
    #print('test: new line positions = ', linePositions)
    #print('test: new transitionStrengths = ', transitionStrengths)
    linesFunc=lambdify((A1,A2,B1,B2), linePositions, modules='numpy')
    #print("linePositions:",[linePositions[i].subs([(A1,AlowerGuess),(A2,AupperGuess),(B1,BlowerGuess),(B2,BupperGuess)])for i in range(len(linePositions))])
    if peakModel=='pseudoVoigt': toAdd = Model(hyperFinePredictionFreeAmps_pseudoVoigt, prefix='iso'+str(k)+'_', independent_vars=['x', 'cec_sim_data', 'equal_fwhm', 'cecBinning'])
    else: toAdd = Model(hyperFinePredictionFreeAmps_voigt, prefix='iso'+str(k)+'_', independent_vars=['x', 'cec_sim_data', 'equal_fwhm', 'cecBinning'])
    params.add('iso'+str(k)+'_'+'iNuc', value=iNuc, vary=False)
    params.add('iso'+str(k)+'_'+'mass', value=mass, vary=False)
    params.add('iso'+str(k)+'_'+'laserFreq', value=laserFreq, vary=False)
    params.add('iso'+str(k)+'_'+'freqOffset', value=freqOffset, vary=False)
    params.add('colinearity', value=colinearity, vary=False) #TODO: understand why these don't have prefixes
    #params.add('cec_sim_data', value=cec_sim_data, vary=False) #TODO: understand why these don't have prefixes
  
    transitionStrengths=np.array(transitionStrengths).astype(float); #print('k=%d; transitionStrengths:'%k,transitionStrengths)
    transitionStrengths*=1/np.max(transitionStrengths); #print(transitionStrengths)
    spFactor = -1 if colinearity else 1
    if cec_sim_data_path==False:
      params.add('iso'+str(k)+'_'+'spProp', value=0.45, vary=True, min=0, max=1)
      params.add('iso'+str(k)+'_'+'spShift', value=spFactor*abs(spShiftGuess), max=max(0, spFactor*200), min=min(0, spFactor*200), vary=(fixed_spShift==False))
    else:
      params.add('iso'+str(k)+'_'+'spProp', value=-1, vary=False)
      params.add('iso'+str(k)+'_'+'spShift', value=-1, vary=False)

    params.add('iso'+str(k)+'_'+'Aupper', value=AupperGuess, vary=not(fixed_Aupper))
    if fixed_Aratio == False: params.add('iso'+str(k)+'_'+'Alower', value=AlowerGuess, vary=not(fixed_Alower))
    else:                     params.add('iso'+str(k)+'_'+'Alower', expr=str(fixed_Aratio)+'*iso'+str(k)+'_'+'Aupper')
    params.add('iso'+str(k)+'_'+'Blower',value=BlowerGuess,vary=(jGround>=1 and iNuc>=1))
    params.add('iso'+str(k)+'_'+'Bupper',value=BupperGuess, vary=(jExcited>=1 and iNuc>=1))
    if k==0:  params.add('iso'+str(k)+'_'+'centroid', value=centroidGuess, vary=True);#value=centroidGuess-spPropGuess/(1+spPropGuess)*spFactor*spShiftGuess
    elif k==1:params.add('iso'+str(k)+'_'+'centroid', value=centroidGuess+20, vary=True);
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
      if peakModel=='pseudoVoigt': params.add('iso'+str(k)+'_'+'alpha', value=0.5,vary=True, min=0, max=1) #note: bring this back for pseudovoigt
    else:
      params.add('iso'+str(k)+'_'+'sigma', expr='iso0_sigma')
      params.add('iso'+str(k)+'_'+'gamma', expr='iso0_gamma')
      if peakModel=='pseudoVoigt': params.add('iso'+str(k)+'_'+'alpha', expr='iso0_alpha') #note: bring this back for pseudovoigt
      params.add('iso'+str(k)+'_'+'spProp', expr='iso0_spProp')
      spFactor = -1 if colinearity else 1
      params.add('iso'+str(k)+'_'+'spShift', expr='iso0_spShift')
    
    refPeakIndex=np.argmax(transitionStrengths) #this is the 0-indexed position of the first peak of maximal racah strength. Rather than adjust the height of this peak, I will fix it at 1 and scale the overall amplitude of the function.
    nominalPositions=[linePositions[i].subs([(A1,AlowerGuess),(A2,AupperGuess),(B1,BlowerGuess),(B2,BupperGuess)])for i in range(len(linePositions))] #locations of lines if initial guess is reasonable
    if k==0: nominalPositions_ground=nominalPositions
    params.add('iso'+str(k)+'_'+'h1', value=ampGuess, vary = True);
    for i in range(12):#len(transitionStrengths)):
      if i+1>len(transitionStrengths):
        for j in range(i,12):
          params.add('iso'+str(k)+'_'+'h'+str(j+1), value=0, vary = False); #print('adding empty peak number %d'%(j+1))
        break
      nomPos=nominalPositions[i]
      varyThisPeak=True
      for j in range(i):
        if abs(nomPos-nominalPositions[j])<(sigmaInit+gammaInit)*1.2:#if nominal position of peak i is separated from nom pos of an earlier peak by less than 100% the linewidth, then the amplitudes of the two will be constrained 
          print('peaks %d and %d are expected to overlap significantly and will be bound to eachother'%((i+1),(j+1)))
          varyThisPeak=False
          params.add('iso'+str(k)+'_'+'h'+str(i+1), expr=str(ampGuess*transitionStrengths[i]/transitionStrengths[j])+'*h'+str(j+1))
      if transitionStrengths[i]<0.01:
        print('test, peak %d, has relative strength of %.3f, and should be held fixed'%((i+1), transitionStrengths[i]))
        params.add('iso'+str(k)+'_'+'h'+str(i+1), expr=str(transitionStrengths[i]/transitionStrengths[0])+'*iso'+str(k)+'_'+'h'+str(1), vary = False, min=0); varyThisPeak=False

      if k!=0:
        for j in range(len(nominalPositions_ground)):
            if abs(nomPos-nominalPositions_ground[j])<(sigmaInit+gammaInit)*.75:
              print('peaks %d and %d are expected to overlap significantly and will be bound to eachother'%((i+1),(j+1)))
              params.add('diff'+str(i+1), value=ampGuess*transitionStrengths[i]/2, vary = True, min=0, max=ampGuess);#TODO: come back to this #value=1
              params.add('iso'+str(k)+'_'+'h'+str(i+1), expr='iso'+str(0)+'_'+'h'+str(j+1)+'- diff'+str(i+1));
              varyThisPeak=False
      if varyThisPeak: 
        params.add('iso'+str(k)+'_'+'h'+str(i+1), value=ampGuess*transitionStrengths[i], vary = True, min=0);

    myMod= myMod + toAdd
  # print('yData', yData, '\n xData:',xData)
  result=myMod.fit(yData, params, x=xData, cec_sim_data=cec_sim_data, equal_fwhm=equal_fwhm, cecBinning=cecBinning, weights=1/yUncertainty, method='leastsq')#, fit_kws={'xtol': 1E-6, 'ftol':1E-6})
  return(result)

def fitAndLogData(mass, targetDirectoryName, iNucList, jGround, jExcited, peakModel='pseudoVoigt', transitionLabel='bruhLabelThis',
  colinearity=True, laserFreq=0, freqOffset=1129900000, energyCorrection=False, centroidGuess=0,
  spShiftGuess=120, cec_sim_data_path=False, fixed_spShift=False, fixed_Alower=False, fixed_Aupper=False, subPath='',
  fixed_Aratio=False, equal_fwhm=False, directoryPrefix='spectralData', weightsList=[2.5,1], fixed_Sigma=False, fixed_Gamma=False,**kwargs):
  #load spectral data from file and run fit
  if subPath!='':
    subPath=subPath.rstrip('/')
    spectrumPath = './'+directoryPrefix+'/mass'+str(round(mass))+'/'+targetDirectoryName+'/'+subPath+'/' #this is where I'll save all outputs of this function
  else: spectrumPath = './'+directoryPrefix+'/mass'+str(round(mass))+'/'+targetDirectoryName+'/' #this is where I'll save all outputs of this function
  if energyCorrection: spectrumPath += '/energyCorrected/' #this is where I'll save all outputs of this function if an energy correction is provided
  if not os.path.exists(spectrumPath): os.makedirs(spectrumPath)

  spectrumFrame = loadSpectrumFrame(mass, targetDirectoryName, energyCorrection=energyCorrection, directoryPrefix=directoryPrefix)
  xData = np.array(spectrumFrame['dcf']);
  yData = np.array(spectrumFrame['countrate']); yUncertainty = np.array(spectrumFrame['uncertainty'])

  #For some reason, when all the frequencies are huge, lmfit struggles to return uncertainties. Subtracting off a recorded offset
  #pre-fit should be a clean way to avoid this issue while still being able to easily recover the true centroid value.
  xData-=freqOffset 
  
  result=fitData(xData, yData, yUncertainty, mass, iNucList, jGround, jExcited, peakModel=peakModel, transitionLabel=transitionLabel, colinearity=colinearity, laserFreq=laserFreq,
                freqOffset=freqOffset, energyCorrection=energyCorrection, centroidGuess=centroidGuess, spShiftGuess=spShiftGuess, cec_sim_data_path=cec_sim_data_path, fixed_spShift=fixed_spShift,
                fixed_Alower=fixed_Alower, fixed_Aupper=fixed_Aupper, subPath=subPath, fixed_Aratio=fixed_Aratio, equal_fwhm=equal_fwhm,  weightsList=weightsList, fixed_Sigma=fixed_Sigma, fixed_Gamma=fixed_Gamma,**kwargs)

  #plotting
  y_fit = result.eval(x=xData)
  x_interp=np.linspace(np.min(xData), np.max(xData), 1000)
  fig, (ax1, ax2) = plt.subplots(2, figsize=(16,9), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

  plotTitle=str(round(mass))+'Al '+targetDirectoryName
  if subPath!='': plotTitle+=' - '+subPath
  plotTitle +=' - colinear' if colinearity else ' - anticolinear'
  plotTitle+=f', fwhm: {equal_fwhm}; cec_sim:{cec_sim_data_path}; fixed_Aratio:{fixed_Aratio}'
  if energyCorrection: plotTitle += ',beam energy corrected by '+str(energyCorrection)
  ax1.set_title(plotTitle)

  #print(result.best_values)
  # print(result.params['bg'].stderr);
  colorList = ['green','red']
  y_interp = result.eval(x=x_interp)
  if len(iNucList)>1:
    components = result.eval_components(x=x_interp)  
    for k,iNuc in enumerate(iNucList): 
      ax1.plot(x_interp, components['iso'+str(k)+'_'], linestyle='--', color=colorList[k], label='nuclear state '+str(k))
  ax1.plot(x_interp, y_interp, 'b-', label='best') #ax1.plot(x_interp, result.model.func(x_interp,**result.best_values), 'g-', label='best')
  #ax1.plot(xData, result.init_fit, 'r--',alpha=0.25, label='init')
  ax1.errorbar(xData,yData,yerr=yUncertainty,fmt='b.',ecolor='black',capsize=1,markersize=8, label='data')
  ax1.set_ylabel('countrate'); ax2.set_ylabel('residuals'); ax2.set_xlabel('(Frequency - ' +str(freqOffset)+')(MHz)')
  ax1.legend(loc=1)
  #ax2.set_title("Residuals")
  ax2.errorbar(xData,(yData-result.best_fit)/yUncertainty,yerr=1,fmt='b.',ecolor='black',capsize=1,markersize=8)
  #plt.show()
  fig.set_size_inches(18.5, 10.5)
  fig.savefig(spectrumPath+'fit_plot.png',dpi=100)
  
  fig.clf()
  plt.close(fig)

  #Writing to file
  statsDic = makeDictionaryFromFitStatistics(result)
  if energyCorrection==False:
    with open(spectrumPath+'fit_report.txt','w') as file:
      file.write(result.fit_report()); file.close()#file.write(result.params.pretty_print()); file.close()
    with open(spectrumPath+'fit_result.pkl','wb') as file:
      pickle.dump(result.params, file)
    with open(spectrumPath+'fit_statistics.pkl','wb') as file:
      pickle.dump(statsDic, file)
  else:
    with open(spectrumPath+'fit_report_energyCorrected.txt','w') as file:
      file.write(result.fit_report()); file.close()#file.write(result.params.pretty_print()); file.close()
    with open(spectrumPath+'fit_result_energyCorrected.pkl','wb') as file:
      pickle.dump(result.params, file)
    with open(spectrumPath+'fit_statistics_energyCorrected.pkl','wb') as file:
      pickle.dump(statsDic, file)

  with open(spectrumPath+'peakPositions.txt','w') as file:
    file.write('Peak positions (MHz):\n')
    for k,iNuc in enumerate(iNucList):
      A1,A2,B1,B2=result.params['iso'+str(k)+'_'+'Alower'].value, result.params['iso'+str(k)+'_'+'Aupper'].value,\
                  result.params['iso'+str(k)+'_'+'Blower'].value, result.params['iso'+str(k)+'_'+'Bupper'].value
      peakFreqs, _ = gimmeLinesAndStrengths(iNuc,jGround,jExcited,A1,A2,B1=B1,B2=B2)
      peakFreqs= np.array(peakFreqs)+freqOffset
      file.write('nuclear state %d: peakFreqs-offset:'%(k+1)+str(peakFreqs-freqOffset)); file.write('\n')
      file.write('centroid: '+str(result.params['iso0_centroid'].value+freqOffset)); file.write('\n')
      for i,peakFreq in enumerate(peakFreqs):
        file.write('peak %d: '%i+str(peakFreq)); file.write('\n')
        if cec_sim_data_path!=False:
          cec_sim_data=np.loadtxt(cec_sim_data_path, skiprows=1,delimiter=',')
          cec_sim_energies = cec_sim_data[:,2]; sp_fractions=cec_sim_data[:,1]
          cec_sim_energies=cec_sim_energies[sp_fractions>0]; sp_fractions=sp_fractions[sp_fractions>0]; originalFractionList=sp_fractions
          sp_scaling_list=result.params['spScaling'].value*np.ones_like(sp_fractions); sp_scaling_list[0]=1
          sp_fractions=sp_fractions*sp_scaling_list
          sp_shifts, sp_fractions, broadeningList = generateSidePeakFreqs(mass, laserFreq, peakFreq, originalFractionList, 
                                                    cec_sim_energies, freqOffset=0, colinearity=colinearity, cecBinning=5)
          file.write('sidepeak frequencies: '+str(sp_shifts)); file.write('\n')
          file.write('sidepeak fractions: '+str(sp_fractions)); file.write('\n')

        else:
          file.write('sidepeak fraction, spacing: '+str(result.params['iso0_spProp'].value)+', '+str(result.params['iso0_spShift'].value)); file.write('\n')
    file.close()

  return(result)

def loadFitResults(directoryPrefix, mass, targetDirectoryName, energyCorrection=False, subPath=''):
  if subPath!='':
    subPath=subPath.rstrip('/')+'/'
  runPath = './'+directoryPrefix+'/mass'+str(round(mass))+'/'+str(targetDirectoryName)+'/'+subPath #this is where I saved all outputs of the earlier function
  if energyCorrection==False: runPath += 'fit_result.pkl' 
  else: runPath += '/energyCorrected/fit_result_energyCorrected.pkl' #this is where I'll save all outputs of this function
  with open(runPath,'rb') as file:
    result=pickle.load(file)
  return(result)

def loadFitStatistics(directoryPrefix, mass, targetDirectoryName, energyCorrection=False, subPath=''):
  if subPath!='':
    subPath=subPath.rstrip('/')+'/'
  runPath = './'+directoryPrefix+'/mass'+str(round(mass))+'/'+str(targetDirectoryName)+'/'+subPath #this is where I saved all outputs of the earlier function
  if energyCorrection==False: runPath += 'fit_statistics.pkl' 
  else: runPath += '/energyCorrected/fit_statistics_energyCorrected.pkl' #this is where I'll save all outputs of this function
  with open(runPath,'rb') as file:
    result=pickle.load(file)
  return(result)

def processData(scanDirec, runs, laserFreq, mass, targetDirectoryName, nuclearSpin, jGround, jExcited, exportSpectrum=False, fitAndLog=True, subPath='', peakModel='pseudoVoigt', fixed_Aratio=False, keepLessIntegratedBins=True,
  colinearity=True, freqOffset=0, tofWindow=[450,550], energyCorrection=False, timeOffset=0, cec_sim_data_path=False, equal_fwhm=False, directoryPrefix='spectralData', fixed_Sigma=False, fixed_Gamma=False,cuttingColumn='time_step', **kwargs):
  print("kwargs:", kwargs)#;quit()
  spectrumPath = './'+directoryPrefix+'/mass'+str(round(mass))+'/'+targetDirectoryName+'/' #this is where I'll save all outputs of this function
  if subPath!='':
    subPath=subPath.rstrip('/')
    fitPath=spectrumPath+subPath+'/'
  else: fitPath = spectrumPath
  if energyCorrection==False: 
    spectrumPath +='spectralData.csv'
    fitPath +='fit_result.pkl'
  else:
    spectrumPath += 'energyCorrected/spectralData_energyCorrected.csv'
    fitPath +='energyCorrected/fit_result_energyCorrected.pkl'
  if not os.path.exists(spectrumPath): exportSpectrum=True; fitAndLog=True #overwrite user if spectral data doesn't already exist
  if not os.path.exists(fitPath): fitAndLog=True; print('flag2', fitPath) #overwrite user if fit doesn't already exist
  if exportSpectrum: 
    print('exporting spectrum for runs '+str(runs))
    exportSpectrumFrame(scanDirec, runs, laserFreq, mass, targetDirectoryName, colinearity=colinearity, windowToF=tofWindow,
    energyCorrection=energyCorrection,timeOffset=timeOffset,directoryPrefix=directoryPrefix, cuttingColumn=cuttingColumn, keepLessIntegratedBins=keepLessIntegratedBins)
  if fitAndLog: fitAndLogData(mass, targetDirectoryName, nuclearSpin, jGround, jExcited, peakModel=peakModel, transitionLabel='P12-S12', colinearity=colinearity,
    laserFreq=laserFreq, freqOffset=freqOffset, energyCorrection=energyCorrection, cec_sim_data_path=cec_sim_data_path, subPath=subPath, 
    fixed_Aratio=fixed_Aratio, equal_fwhm=equal_fwhm,directoryPrefix=directoryPrefix,fixed_Sigma=fixed_Sigma, fixed_Gamma=fixed_Gamma, **kwargs)

def populateFrame(mass, mass_uncertainty, nuclearSpin, result, prefix='iso0', fixed_Aratio=False, uncorrectedResult=False, scanTime=False):
  aLower=result[prefix+'_Alower'].value; aLower_uncertainty=result[prefix+'_Alower'].stderr
  aUpper=result[prefix+'_Aupper'].value; aUpper_uncertainty=result[prefix+'_Aupper'].stderr
  aRatio=aLower/result[prefix+'_Aupper'].value
  aRatio_uncertainty = (aRatio**2) *( (aLower_uncertainty/aLower)**2 + (aUpper_uncertainty/aUpper)**2)
  print(aRatio, aRatio_uncertainty)
  tempDict={
  'massNumber':round(mass),
  'mass':[mass],
  'mass_uncertainty':[mass_uncertainty],
  "I":[nuclearSpin],
  "aLower":[aLower],
  "aLower_uncertainty":[aLower_uncertainty],
  "aUpper":[aUpper],
  "aUpper_uncertainty":[aUpper_uncertainty],
  "aRatio":[aRatio],
  "aRatio_uncertainty":[0] if fixed_Aratio else [aRatio_uncertainty],
  "centroid":[result[prefix+'_centroid'].value],
  "cent_uncertainty":[result[prefix+'_centroid'].stderr]}#,None,None,None,None]]
  if uncorrectedResult != False:
    tempDict["uncorrectedCentroid"]=[uncorrectedResult[prefix+'_centroid'].value]
    tempDict["uncorrectedCentroid_uncertainty"]=[uncorrectedResult[prefix+'_centroid'].stderr]
  if scanTime != False:
    tempDict["avgScanTime"]=[scanTime]

  return(pd.DataFrame(tempDict))

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

def generateSidePeakFreqs(mass, laserFreq, peakFreq, sp_fractions, cec_sim_list, freqOffset=0, colinearity=True, cecBinning=False):
  '''
  based on input mass, laser frequency, and resonance frequency, determine resonance in voltage space, then 
  apply e_losses from cec_sim_list, and finally transform back to frequency space to predict sidepeak positions
  '''
  #print(sp_fractions)
  originalPeakFreq=peakFreq#...this seems to change slightly if I don't do this
  cec_sim_list=cec_sim_list[sp_fractions!=0] #removing zero-amplitude entries from e_loss list
  sp_fractions=sp_fractions[sp_fractions!=0] #removing zero-amplitude entries from fraction list

  cec_sim_list[cec_sim_list<0] /=3 #update May 23, 2025: dividing energy loss channels by 3, for some reasons.

  neutralRestEnergy=mass*amu2eV
  ionRestEnergy=neutralRestEnergy-electronRestEnergy

  peakFreq+=freqOffset #need to use actual resonance frequency, not fitting frequency (which is offset), or else Doppler transforms are wrong
  if colinearity:
    beta_0 = (peakFreq**2-laserFreq**2)/(peakFreq**2+laserFreq**2); #print(beta_0);
  else:
    beta_0 = (peakFreq**2-laserFreq**2)/(peakFreq**2+laserFreq**2); #print(beta_0); 
  gamma_0=1/np.sqrt(1-beta_0**2); #print(gamma_0)
  voltage_0 = ionRestEnergy*(gamma_0-1); #print(voltage_0)
  voltageList = voltage_0 - cec_sim_list; ##e-losses actually look like higher voltages, because you had to accelerate more to end up resonant after inelastic collision 
  betaList=np.sqrt(1-((ionRestEnergy)/(voltageList+ionRestEnergy))**2)
  if colinearity:
    dcfList = np.sqrt(1-betaList)/np.sqrt(1+betaList)*laserFreq #doppler corrected frequencies
  else:
    dcfList = np.sqrt(1+betaList)/np.sqrt(1-betaList)*laserFreq #doppler corrected frequencies
  dcfList-=freqOffset

  sp_shifts=np.zeros(len(dcfList[cec_sim_list<0])+1)
  
  sp_shifts[1:]=dcfList[cec_sim_list<0]
  sp_shifts[0] = originalPeakFreq#peakFreq-freqOffset
  negativeOffsets=dcfList[cec_sim_list>0]-sp_shifts[0]
  negOffsetWeights=sp_fractions[cec_sim_list>0]
  #broadeningList=np.zeros(len(dcfList[cec_sim_list<0])+1)
  broadeningFactor = np.sum(negativeOffsets*negOffsetWeights)/np.sum(negOffsetWeights)#question: should I sum the positive e_losses linearly or in quadrature? TODO: Confirm that it makes sense to transform to MHz

  if cecBinning!=False: #if true, we are binning the simulated sidepeaks in bins of width cecBinning, and we need to calculate weighted means for sp_positions, and sum the appropriate fractiosn 
    bins=np.arange(np.min(sp_shifts),np.max(sp_shifts)+cecBinning, cecBinning)#-cecBinning/2
    bindices = np.digitize(sp_shifts,bins)
    finalLength=len(np.unique(bindices))
    tempShifts=np.zeros(finalLength); tempFracts=np.zeros(finalLength)

    for i, bindex in enumerate(np.unique(bindices)):
      tempFracts[i] = np.sum(sp_fractions[bindices==bindex])
      tempShifts[i] = np.sum(sp_fractions[bindices==bindex]*sp_shifts[bindices==bindex])/tempFracts[i]
    sp_shifts=tempShifts
    sp_fractions=tempFracts
  broadeningList=np.zeros(len(sp_shifts)); broadeningList[0] = broadeningFactor
  sp_shifts-=sp_shifts[0]# I'm only returning the shifts to the main peak frequency to make the use case slightly more general

  return(sp_shifts, sp_fractions, broadeningList)

if __name__ == '__main__': pass