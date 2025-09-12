import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
from lmfit import model
import os
import time
from numba import njit

amu2eV = np.float64(931494102.42)
electronRestEnergy=510998.950 #in eV

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

def cutToF(df, minT, maxT, cuttingColumn='time_step',**kwargs):
  print('cuttingColumn: ', cuttingColumn)
  df=df.filter(pl.col(cuttingColumn) > minT).filter(pl.col(cuttingColumn) < maxT)
  return(df)

def makeSpectrum(dataFrame, laserFreq, mass, colinear=True, **kwargs):
  #Bins events by total Voltage, converts to frequency, returns dataframe that can be treated as spectrum
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
  groupFrame=dataFrame.group_by('time_step')
  # tData=np.array(dataFrame.group_by('time_step').agg(pl.col('ToF').mean()))
  yData=np.array(groupFrame.agg(pl.col('PMT0').sum()))
  order=np.argsort(yData[:,0])
  d={'tof':yData[order,0],'counts':yData[order,1]}
  spectrumFrame=pd.DataFrame(data=d)
  return(spectrumFrame)

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
  if not os.path.exists(spectrumPath):
    os.makedirs(spectrumPath)
  #plot ToF spectrum, and make cuts if input
  fig=plt.figure()
  t2=time.perf_counter()
  timeSpec=tofSpectrum(polarsFrame)
  t3=time.perf_counter()
  plt.plot(timeSpec['tof'],timeSpec['counts'])
  if len(windowToF)==2:
    #print('making tof cut: ['+str(windowToF[0])+', '+str(windowToF[1])+']')
    # dFrame=cutToF(dFrame, windowToF[0], windowToF[1],**kwargs); timeSpec=tofSpectrum(dFrame)
    polarsFrame=cutToF(polarsFrame, windowToF[0], windowToF[1],**kwargs); timeSpec=tofSpectrum(polarsFrame)
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

def loadSpectrumFrame(mass, targetDirectoryName, energyCorrection=False, directoryPrefix='spectralData'):
  spectrumPath = './'+directoryPrefix+'/mass'+str(round(mass))+'/'+targetDirectoryName+'/' #this is where I'll save all outputs of this function
  if energyCorrection:
    spectrumPath += '/energyCorrected/' #this is where I'll save all outputs of this function if an energy correction is provided
    spectrumFrame = pd.read_csv(spectrumPath+'spectralData_energyCorrected.csv')
  else: spectrumFrame = pd.read_csv(spectrumPath+'spectralData.csv')
  return(spectrumFrame)