import os
import time
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lmfit import Model,Parameters, Parameter, model

def getDataTime(filename):
  file = tuple(open(filename,'r'))
  line=file[60]
  #print(line)
  if "# Scan time =" in line:
    timeString=line.strip('# Scan time =\n')
    result = time.mktime(time.strptime(timeString, '%B %d, %Y  %H:%M:%S.%f' ))
  else:
    for line in file:
      if "# Scan time =" in line:
        timeString=line.strip('# Scan time =\n')
        result = time.mktime(time.strptime(timeString, '%B %d, %Y  %H:%M:%S.%f' ))
        break
  return(result)

def import1DScan(filename, energyCorrection=False,
  cols = [0,1,2,4,8,11,13,15,16,17,18,19],
  colNames=['run','region','vstep','time_step','scan_volt_set','scan_volt_read','HV_read','laserFreq','ToF','PMT0','PMT1','PMT2']):
  '''this function reads a single .asc file, which represents a full ToF window at a single voltage step, during a single run of a scan
  The function creates a pandas dataframe of all of the relevant information, including the time at which the data was recorded'''
  
  file=open(filename)
  reader = csv.reader(file, delimiter=r' ', skipinitialspace=True)
  skip = 0
  for row in reader:
    if len(row)<20:
      skip +=1
    else: break
  file.close()
  if skip!=100:  print('hmmm...', skip)
  dFrame = pd.read_csv(filename, comment='#', delimiter=r'\s+',usecols=cols,names=colNames,skiprows=skip)
  dFrame["scan_volt_set"]=dFrame["scan_volt_set"]*1000
  dFrame["scan_volt_read"]=dFrame["scan_volt_read"]*201.0037
  dFrame["HV_read"]=dFrame["HV_read"]*2980.32
  dFrame['totalVoltage'] = dFrame['HV_read'] - dFrame['scan_volt_read']
  scanTime=getDataTime(filename);  dFrame['scanTime']=scanTime
  if energyCorrection!=0:
    dFrame['totalVoltage'] += energyCorrection
  dFrame['toCount']=np.ones_like(dFrame['totalVoltage'])
  
  return(dFrame)

def readScanToCSV(dataDirectory):
  dFrame=pd.DataFrame()
  scanFilenames=filenames = os.listdir(dataDirectory)
  for fname in scanFilenames:
    print('fname:', fname)
    if '.asc' in fname:
      tempFrame=import1DScan(dataDirectory+fname)
      dFrame=pd.concat([dFrame,tempFrame])
  return(dFrame)

def mda2csv(dataDirectory):
  dataDirectory=dataDirectory.strip('/\\')+'/'
  filenames = os.listdir(dataDirectory)
  for fname in filenames:
    if '.mda' in fname:
      subdir = dataDirectory+'scan'+fname.lstrip('DBEC_').rstrip('.mda')+'/';
      saveFileName=subdir+'scan'+fname.lstrip('DBEC_').rstrip('.mda')+'_DataFrame.csv'
      try:
        os.mkdir(subdir)
        os.system("mda2ascii.exe -fm -i1 -o "+subdir+' '+dataDirectory+fname)
        print('success')
      except: pass
      print(fname); print(subdir)
      try: totalFrame=pd.read_csv(saveFileName)
      except:
        totalFrame=readScanToCSV(subdir);#print(totalFrame)
        totalFrame.to_csv(saveFileName)
      print(saveFileName)

def importDataFrame(path, scan):
  #loads csv-converted form of raw scan data
  filename=path+'/scan%d/scan%d_DataFrame.csv'%(scan,scan)
  dFrame = pd.read_csv(filename)
  return(dFrame)

def cutToF(dataFrame, minT, maxT, cuttingColumn='time_step'):
  print('cuttingColumn: ', cuttingColumn)
  dataFrame2=dataFrame.copy()
  dataFrame2[cuttingColumn]=dataFrame2[cuttingColumn].map(lambda v: v if (v>minT and v<maxT) else float('NaN'))
  dataFrame2.dropna(inplace=True)
  return(dataFrame2)

def makeSpectrum(dataFrame, laserFreq, mass, colinear=True, windowToF=[], **kwargs):
  #TODO
  electronRestEnergy=510998.950 #in eV
  amu2eV = np.float64(931494102.42)
  neutralRestEnergy=mass*amu2eV
  ionRestEnergy=neutralRestEnergy-electronRestEnergy

  if len(windowToF)==2:
    minToF=windowToF[0]; maxToF=windowToF[1]
    dataFrame = cutToF(dataFrame, minToF, maxToF, **kwargs)
  dataFrame['partialWeights']=dataFrame['totalVoltage']*dataFrame['PMT0']
  vSteps=np.max(dataFrame['vstep']) #use this to decide how many bins to make
  tSteps=np.max(dataFrame['time_step'])-np.min(dataFrame['time_step'])+1 #this represents the number of "toCount" entries I would get from a single measurement at a given vStep
  vCuts = pd.cut(dataFrame.loc[:,'scan_volt_set'], bins=vSteps, retbins=False) #bin on 'scan_volt_set' in case of different scan starting points
  groupFrame=dataFrame.groupby(vCuts, observed=False) #I don't think this observed keyword is important, but it shuts up a FutureWarning
  
  voltageData=np.array(groupFrame.mean()['totalVoltage']); #TODO: Switch this back to weighted average!
  noZerosMask=np.array(groupFrame.sum()['PMT0'])!=0
  voltageData[noZerosMask]=np.array(groupFrame.sum()['partialWeights'])[noZerosMask]/np.array(groupFrame.sum()['PMT0'])[noZerosMask] #count-weighted mean of voltages probed in each vstep grouping
  # voltageData=np.array(groupFrame.sum()['partialWeights'])/np.array(groupFrame.sum()['PMT0']) #count-weighted mean of voltages probed in each vstep grouping
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
  energyCorrection=False, timeOffset=0, directoryPrefix='spectralData', keepLessIntegratedBins=False,**kwargs):
  dFrame=pd.DataFrame()
  for run in runs:
    print('loading run %d'%run)
    currentDataFrame=importDataFrame(scanDirec, run)
    dFrame=pd.concat([dFrame,currentDataFrame])
  
  spectrumPath = './'+directoryPrefix+'/mass'+str(round(mass))+'/'+targetDirectoryName+'/' #this is where I'll save all outputs of this function
  if energyCorrection: 
    spectrumPath += 'energyCorrected/' #this is where I'll save all outputs of this function if an energy correction is provided
    if not os.path.exists(spectrumPath): os.makedirs(spectrumPath)
    if type(energyCorrection)==model.ModelResult:
      scanTimes=np.array(dFrame['scanTime']) - timeOffset      
      voltageCorrections = energyCorrection.eval(x=scanTimes); print('test:voltageCorrections:\n',voltageCorrections)
      print('Hot dog, using calibration function for energy correction!')
      # np.savetxt(spectrumPath+'voltageCorrections.csv', np.c_[dFrame['scanTime'][::1024],voltageCorrections[::1024]], delimiter=',') #this is huge lmao
      with open(spectrumPath+'averageEnergyCorrection.txt','w') as file:
        file.write('<Delta E(t)>='+str(np.mean(voltageCorrections))+'V\tRange: '+str([np.min(voltageCorrections), np.max(voltageCorrections)])); file.close()
      print('test:scan times:\n',scanTimes)

    else: voltageCorrections = energyCorrection #presumably this is a float or int potentially
    try: dFrame['totalVoltage']+= voltageCorrections
    except Exception as e: print(e.dir()); quit()

  if not os.path.exists(spectrumPath):
    os.makedirs(spectrumPath)
  print("test: len(dFrame)=",len(dFrame), 'windowToF=',windowToF)
  #plot ToF spectrum, and make cuts if input
  fig=plt.figure()
  timeSpec=tofSpectrum(dFrame)
  plt.plot(timeSpec['tof'],timeSpec['counts'])
  if len(windowToF)==2:
    #print('making tof cut: ['+str(windowToF[0])+', '+str(windowToF[1])+']')
    dFrame=cutToF(dFrame, windowToF[0], windowToF[1],**kwargs)
    timeSpec=tofSpectrum(dFrame)    
    plt.plot(timeSpec['tof'],timeSpec['counts'])
  plt.title(str(round(mass))+'Al ToF Histogram - '+targetDirectoryName)
  fig.set_size_inches(18.5, 10.5)
  fig.savefig(spectrumPath+'tof_plot.png',dpi=200)
  fig.clf()
  plt.close(fig)

  fig=plt.figure()
  print("test: len(dFrame)=",len(dFrame))
  #make spectrum, write it to a file, and then plot it
  spectrumFrame = makeSpectrum(dFrame, laserFreq, mass, colinear=colinearity)
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

def loadSpectrumFrame(mass, targetDirectoryName, energyCorrection=False, directoryPrefix='spectralData'):
  spectrumPath = './'+directoryPrefix+'/mass'+str(round(mass))+'/'+targetDirectoryName+'/' #this is where I'll save all outputs of this function
  if energyCorrection:
    spectrumPath += '/energyCorrected/' #this is where I'll save all outputs of this function if an energy correction is provided
    spectrumFrame = pd.read_csv(spectrumPath+'spectralData_energyCorrected.csv')
  else: spectrumFrame = pd.read_csv(spectrumPath+'spectralData.csv')
  return(spectrumFrame)

if __name__ == '__main__':
  mda2csv('25Al')