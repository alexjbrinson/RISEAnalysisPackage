import os
import time
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import polars as pl

def getDataTime(filename, timeLineNumber=60,**kwargs):
  file = tuple(open(filename,'r'))
  line=file[timeLineNumber]
  #print(line)
  if "# Scan time =" in line:
    timeString=line.strip('# Scan time =\n')
    result = time.mktime(time.strptime(timeString, '%b %d, %Y  %H:%M:%S.%f' ))
  else:
    for line in file:
      if "# Scan time =" in line:
        timeString=line.strip('# Scan time =\n')
        result = time.mktime(time.strptime(timeString, '%B %d, %Y  %H:%M:%S.%f' ))
        break
  return(result)

def import1DScan(filename, energyCorrection=False, timeLineNumber=60,
                 cols = [0,1,2,4,8,11,13,15,16,17,18,19],
                 colNames=['run','region','vstep','time_step','scan_volt_set','scan_volt_read','HV_read','laserFreq','ToF','PMT0','PMT1','PMT2'],
                 **kwargs): #windowToF=[]
  '''this function reads a single .asc file, which represents a full ToF window at a single voltage step, during a single run of a scan
  The function creates a pandas dataframe of all of the relevant information, including the time at which the data was recorded'''
  
  assert(len(cols)==len(colNames))
  file=open(filename)
  reader = csv.reader(file, delimiter=r' ', skipinitialspace=True)
  skip = 0
  for row in reader:
    if len(row)<20:
      skip +=1
    else: break
  file.close()
  if skip>100:  print('hmmm...', skip)
  scanTime=getDataTime(filename, timeLineNumber=timeLineNumber)
  #unfortunately, reading the raw.ascii with polars is painful bc separator can't be reg-ex, and the .ascii is delimited by variable number of spaces.
  dFrame = pd.read_csv(filename, comment='#', delimiter=r'\s+',usecols=cols,names=colNames,skiprows=skip)
  dFrame["scan_volt_set"]=dFrame["scan_volt_set"]*1000
  dFrame["scan_volt_read"]=dFrame["scan_volt_read"]*201.0037
  dFrame["HV_read"]=dFrame["HV_read"]*2980.32
  dFrame['totalVoltage'] = dFrame['HV_read'] - dFrame['scan_volt_read']
  scanTime=getDataTime(filename);  dFrame['scanTime']=scanTime
  if energyCorrection!=0:
    dFrame['totalVoltage'] += energyCorrection
  dFrame['toCount']=np.ones_like(dFrame['totalVoltage'])
  dFrame=pl.from_pandas(dFrame)
  return(dFrame)

def readScanToCSV(dataDirectory,**kwargs):
  dFrame=pl.DataFrame()
  scanFilenames=filenames = os.listdir(dataDirectory)
  updateInterval=int(len(scanFilenames)/100); progress=0
  t0=time.perf_counter()
  for i,fname in enumerate(scanFilenames):
    # if i%updateInterval==0: progress+=1; print(f"progress:{progress}%")
    # print('fname:', fname)
    if '.asc' in fname:
      tempFrame=import1DScan(dataDirectory+fname,**kwargs)
      dFrame=pl.concat([dFrame,tempFrame])
  t1=time.perf_counter()
  print(f'time elapsed: {t1-t0}')
  return(dFrame)

def main(dataDirectory, **kwargs):
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
      # try: totalFrame=pd.read_csv(saveFileName)
      # except:
      totalFrame=readScanToCSV(subdir, **kwargs);#print(totalFrame)
      totalFrame.write_csv(saveFileName)
      print(saveFileName)

if __name__ == '__main__':
  main('27Al')