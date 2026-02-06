import hyperfinePredictorGREAT as hpg
import SpectrumClass as spc
import pandas as pd
import numpy as np
import sympy
import os
import pickle
import matplotlib.pyplot as plt
from lmfit.models import LinearModel

def getCalibrationFunction(v0, δv0, calibrationFrame, xData, mass, laserFreq, randomSampling=False):
  yData=[]; yerr=[]

  for run in calibrationFrame.index:
    fa = calibrationFrame.loc[run,'centroid']; δfa = calibrationFrame.loc[run,'cent_uncertainty']
    if randomSampling:
      fa = np.random.normal(loc=fa, scale=δfa) #random sample for centroid frequency based on fit statistics
      ΔEkin =calculateBeamEnergyCorrectionFromv0vc(mass, laserFreq, fa, v0); yData+=[ΔEkin]
      δΔEkin=propagateBeamEnergyCorrectionUncertainties([mass,0], [laserFreq,1], [fa, δfa], [v0,δv0]); yerr+=[δΔEkin]
    else:
      ΔEkin =calibrationFrame.loc[run,'ΔEkin']; yData+=[ΔEkin]
      δΔEkin=calibrationFrame.loc[run,'ΔEkin_uncertainty']; yerr+=[δΔEkin]
  res = LinearModel().fit(np.array(yData), x=xData, slope=1, intercept=0, weights=1/np.array(yerr), method='leastsq', fit_kws={'xtol': 1E-6, 'ftol':1E-6})
  return(res)

def updateBeamEnergyCorrections(mass, colinearRuns, anticolinearRuns, laserDic, compCoParmResults, compAntiParmResults,δlaserFreq=1):
  v0_estimates=[]
  energyCorrectionList=[]
  laserFreqsCo=np.array([laserDic[run] for run in colinearRuns]); uniqueLaserFreqsCo=np.unique(laserFreqsCo)
  laserFreqsAnti=np.array([laserDic[run] for run in anticolinearRuns]); uniqueLaserFreqsAnti=np.unique(laserFreqsAnti)
  
  for lfc in uniqueLaserFreqsCo:
    coRuns = colinearRuns[laserFreqsCo == lfc]
    #print('colinearRuns: ',colinearRuns)
    colinearCentroidResults = [list(compCoParmResults.loc[run,['centroid', 'cent_uncertainty']]) for run in coRuns]
    colinearWeightedStats=weightedStats(*np.array(colinearCentroidResults).transpose())
    fc=colinearWeightedStats[0]; #print('lfc:',lfc,'colinearWeightedStats:',colinearWeightedStats)
    δfc = np.sqrt(colinearWeightedStats[1]**2+colinearWeightedStats[2]**2)
    for lfa in uniqueLaserFreqsAnti:
      antiRuns=anticolinearRuns[laserFreqsAnti==lfa]
      #print('anticolinearRuns: ',antiRuns)
      anticolinearCentroidResults = [list(compAntiParmResults.loc[run,['centroid', 'cent_uncertainty']]) for run in antiRuns]
      anticolinearWeightedStats=weightedStats(*np.array(anticolinearCentroidResults).transpose())
      fa=anticolinearWeightedStats[0];                                                                            
      δfa = np.sqrt(anticolinearWeightedStats[1]**2+anticolinearWeightedStats[2]**2)
      ΔEkin, centroidEstimate=calculateBeamEnergyCorrection(mass, lfc,lfa,fc, fa)
      print('test?', lfc, lfa, ΔEkin)
      energyCorrectionList+=[ΔEkin]
      v0_estimates+=[bootstrapUncertainty(get_v0,2000,[ [mass,0],[lfc,δlaserFreq],[lfa,δlaserFreq],[fc,δfc],[fa,δfa] ])]
      
    # generating a bootstrapping convergence plot
    # convergenceArray=[]
    # nSamples = np.array(list(range(2,20)) + list(range(20,5000,10))); print(nSamples)
    # for n in nSamples:
    #   print(n)
    #   convergenceArray+=[bootstrapUncertainty(get_v0, n,[ [mass,0],[lfc,δlaserFreq],[lfa,δlaserFreq],[fc,δfc],[fa,δfa] ])[1]]
    # plt.plot(nSamples, convergenceArray)
    # plt.xlabel('num Samples'); plt.ylabel(r'$\sigma_{v_0}$'); plt.title('Convergence of centroid error estimate from boostrapping')
    # plt.show()

  v0_final = weightedStats(*np.array(v0_estimates).transpose())
  print("v0_estimatesList:\n",v0_estimates,"\nv0_final:",v0_final) 
    
  return(v0_final)

def updateLaserDic(logDirectory):
  laserFreqDic={}
  logFileList=os.listdir(logDirectory)
  for logFile in logFileList:
    if '.xlsx' in logFile:
      print('importing log file ', logFile)
      logFrame=pd.read_excel(f'{logDirectory}/{logFile}',usecols=['Run #', 'Laser Freq. (THz)']).dropna()
      for index, row in logFrame.iterrows():
        laserFreqDic[int(row['Run #'])]=row['Laser Freq. (THz)']
  with open('laserDic.pkl','wb') as file: pickle.dump(laserFreqDic, file); file.close()

def weightedStats(values, errors):
  try: 
    sigmaPropagated=np.sqrt(len(errors))/np.sum(1/errors); weightArray=(1/errors); weightedMean=np.sum(weightArray*values)/np.sum(weightArray)
  # try: sigmaPropagated=np.sqrt(1/np.sum(1/errors**2)); weightArray=1/errors; weightedMean=np.sum(weightArray*values)/np.sum(weightArray)
  except: sigmaPropagated=float('NaN'); weightedMean=float('NaN')

  return(weightedMean, sigmaPropagated, np.std(values)/np.sqrt(len(values)))

def calculateBeamEnergyCorrection(mass, vc, va, fc, fa):
  #TODO: what mass does this assume?
  electronRestEnergy=510998.950 #in eV
  amu2eV = np.float64(931494102.42)
  neutralRestEnergy=mass*amu2eV
  ionRestEnergy=neutralRestEnergy-electronRestEnergy
  v0=np.sqrt(vc*va); v0old=v0
  Uc = ( (1 - ( (vc**2-fc**2)/(vc**2+fc**2) )**2)**(-1/2) - 1) *mass*amu2eV
  Ua = ( (1 - ( (va**2-fa**2)/(va**2+fa**2) )**2)**(-1/2) - 1) *mass*amu2eV
  ΔU = Uc-Ua;# print('ΔU=',ΔU,'eV')
  sens_c=(2*v0)/(ionRestEnergy)*(vc**2)/(vc**2-v0**2); sens_a=(2*v0)/(ionRestEnergy)*(va**2)/(va**2-v0**2) 
  v0=np.sqrt( (vc-sens_c*ΔU)*va )
  while np.abs(v0 - v0old)>.001:
    v0old=v0
    sens_c=(2*v0)/(mass*amu2eV)*(vc**2)/(vc**2-v0**2);
    v0=np.sqrt( (vc-sens_c*ΔU)*va )
  ΔEkin = -(ionRestEnergy)*(v0-fc)*(vc**2-v0**2)/(2*v0*vc**2)#Using eq 2.13
  #print("ΔEkin = ", ΔEkin)
  return(ΔEkin, v0)

def calculateBeamEnergyCorrectionFromv0vc(mass, vc, fc, v0):
  #TODO: what mass does this assume?
  electronRestEnergy=510998.950 #in eV
  amu2eV = np.float64(931494102.42)
  neutralRestEnergy=mass*amu2eV
  ionRestEnergy=neutralRestEnergy-electronRestEnergy
  ΔEkin = -(ionRestEnergy)*(v0-fc)*(vc**2-v0**2)/(2*v0*vc**2)#Using eq 2.13
  #print("ΔEkin = ", ΔEkin)
  return(ΔEkin)

def propagateBeamEnergyCorrectionUncertainties(massTuple, vcTuple, fcTuple, v0Tuple):
  mass, vc, fc, v0 = massTuple[0], vcTuple[0], fcTuple[0], v0Tuple[0]
  δmass, δvc, δfc, δv0 = massTuple[1], vcTuple[1], fcTuple[1], v0Tuple[1]
  amu2eV = np.float64(931494102.42)

  dEdvm = -(amu2eV)*(v0-fc)*(vc**2-v0**2)/(2*v0*vc**2)
  dEdvc = amu2eV*mass*v0*(fc-v0)/(vc**3)
  dEdfc = amu2eV*mass*(vc**2-v0**2)/(2*v0*(vc**2))
  dEdv0 = amu2eV*mass*(2*v0**3-fc*(v0**2+vc**2))/(2*(v0**2)*(vc**2))

  δΔE_squared = (abs(dEdvm)**2)*(δmass**2) + (abs(dEdvc)**2)*(δvc**2) + (abs(dEdfc)**2)*(δfc**2) + (abs(dEdv0)**2)*(δv0**2)
  δΔE=np.sqrt(δΔE_squared)
  return(δΔE)

def bootstrapUncertainty(func, nTrials, parmValErrorPairsList):
    outputs=[func(*[ np.random.normal(loc=parmValErrorPairsList[j][0], scale=parmValErrorPairsList[j][1]) for j in range(len(parmValErrorPairsList))]) for i in range(nTrials)]
    #print('standard deviation from %dtrials: '%nTrials, np.std(outputs) )
    sigma = np.std(outputs)
    x0 = func(*[ parmValErrorPairsList[j][0] for j in range(len(parmValErrorPairsList))])
    #print('comparison for curiosity" x0 = %.3f ; xbar = %.3f'%(x0, np.mean(outputs)))
    return([x0, sigma])

def get_v0(mass, vc, va, fc, fa): return(calculateBeamEnergyCorrection(mass, vc, va, fc, fa)[1])

def analyzeTransition(colinearRuns, anticolinearRuns, laserDic, redoFits=False,
                      eCorrectionsForRun=False, spectrumKwargs={}, fittingKwargs={}):
  skwargsCopy=spectrumKwargs.copy() #dictionaries passed by reference. If I don't make copy, I overwrite target directory root
  targetDirectoryRoot=skwargsCopy['targetDirectory']
  compiledColinearParmResults=pd.DataFrame()
  compiledAnticolinearParmResults=pd.DataFrame()
  for run in colinearRuns:
    colinearity=True
    eco=False if eCorrectionsForRun==False else [eCorrectionsForRun[run]]
    skwargsCopy['targetDirectory']=targetDirectoryRoot+'/Colinear/Scan%d'%run
    skwargsCopy.update({'runs':[run], 'colinearity':colinearity, 'laserFrequency':laserDic[run], 'energyCorrection':eco})
    spec=spc.Spectrum(**skwargsCopy);
    spec.fitAndLogData(**fittingKwargs)
    popFrame=spec.populateFrame(prefix="iso0",index=run); compiledColinearParmResults=pd.concat([compiledColinearParmResults, popFrame])

  for run in anticolinearRuns:
    colinearity=False
    eco=False if eCorrectionsForRun==False else [eCorrectionsForRun[run]]
    skwargsCopy['targetDirectory']=targetDirectoryRoot+'/Anticolinear/Scan%d'%run
    skwargsCopy.update({'runs':[run], 'colinearity':colinearity, 'laserFrequency':laserDic[run], 'energyCorrection':eco})
    spec=spc.Spectrum(**skwargsCopy); spec.fitAndLogData(**fittingKwargs)
    popFrame=spec.populateFrame(prefix="iso0",index=run); compiledAnticolinearParmResults=pd.concat([compiledAnticolinearParmResults, popFrame])

  return(compiledColinearParmResults, compiledAnticolinearParmResults)

def getEnergyCorrectedResults(colinearRuns, anticolinearRuns, laserDic, redoFits=False, spectrumKwargs={}, fittingKwargs={},
                              redoFitWithEnergyCorrection=False):
  mass=spectrumKwargs['mass']
  colinearRuns=np.array(colinearRuns); anticolinearRuns=np.array(anticolinearRuns)
  compiledColinearParmResults, compiledAnticolinearParmResults = analyzeTransition(colinearRuns, anticolinearRuns, laserDic, 
                                                                                   spectrumKwargs=spectrumKwargs,fittingKwargs=fittingKwargs,
                                                                                   redoFits=redoFits, eCorrectionsForRun=False)
  
  correctedCentroidEstimate = updateBeamEnergyCorrections(mass, colinearRuns, anticolinearRuns, laserDic, compiledColinearParmResults, compiledAnticolinearParmResults)

  if redoFitWithEnergyCorrection:
    centroids=pd.concat([compiledColinearParmResults,compiledAnticolinearParmResults])['centroid']
    energyCorrectionForRun={run: calculateBeamEnergyCorrectionFromv0vc(mass, centroids[run], laserDic[run], correctedCentroidEstimate[0]) for run in centroids.keys()}
    compiledColinearParmResults_Corrected, compiledAnticolinearParmResults_Corrected = analyzeTransition(colinearRuns, anticolinearRuns, laserDic,
                                                                                                         spectrumKwargs=spectrumKwargs,fittingKwargs=fittingKwargs,
                                                                                                         redoFits=redoFits, eCorrectionsForRun=energyCorrectionForRun)
  
    return(correctedCentroidEstimate, compiledColinearParmResults, compiledAnticolinearParmResults,compiledColinearParmResults_Corrected, compiledAnticolinearParmResults_Corrected)
  return(correctedCentroidEstimate, compiledColinearParmResults, compiledAnticolinearParmResults, -1, -1)
        

def main(cec_sim_data_path=False, equal_fwhm=False, redoFitWithEnergyCorrection=True, redoFits=False):

  tofWindow=[489,543]
  scanDirec='Anti_Colinear_Data'; logDirec=scanDirec; updateLaserDic(logDirec)
  with open('laserDic.pkl','rb') as file: laserDic=pickle.load(file); file.close()
  for key in laserDic.keys(): laserDic[key]*=3E6 #conversion to MHz, and frequency tripled output

  targetDirectoryName='beamEnergy_analysis'
  mass=26.98153841; 
  iNuc27  =2.5; jElecP12=0.5; jElecS12=0.5
  directoryPrefix='results'+'/equal_fwhm_'+str(equal_fwhm)+'/cec_sim_toggle_'+str(cec_sim_data_path!=False)
  spectrumKwargs={'mass':mass,'jGround':jElecP12, 'jExcited':jElecS12, 'nuclearSpinList':[iNuc27],
                  'directoryPrefix':directoryPrefix,'targetDirectory':targetDirectoryName, 'scanDirectory':scanDirec,
                  'windowToF':tofWindow,'cuttingColumn':'time_step', 'constructSpectrum':False}
  fittingKwargs ={'cec_sim_data_path':cec_sim_data_path,'equal_fwhm':equal_fwhm, 'peakModel':'pseudoVoigt','transitionLabel':'P12-S12'}
  anticolinearRuns = [16253,16254,16255,16263,16264,16265]
  colinearRuns     = [16258,16259,16260,16268,16269,16270]

  v0_final, _,_,_,_ = getEnergyCorrectedResults(colinearRuns, anticolinearRuns, laserDic, spectrumKwargs=spectrumKwargs, fittingKwargs=fittingKwargs,
                                                redoFits=redoFits, redoFitWithEnergyCorrection=redoFitWithEnergyCorrection)
  print(v0_final[0])
  assert(abs(v0_final[0]-1129899838)<0.3) #(sanity check while I refactor)
  print(abs(v0_final[0]-1129899838)<0.3)
  return(v0_final)

if __name__==  '__main__': 
  main(cec_sim_data_path=False, equal_fwhm=True, redoFitWithEnergyCorrection=True)