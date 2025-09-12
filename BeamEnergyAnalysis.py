import hyperfinePredictorGREAT as hpg
import SpectrumClass as spc
import pandas as pd
import numpy as np
import sympy
import os
import pickle
import matplotlib.pyplot as plt
from lmfit.models import LinearModel

def getCalibrationFunction(v0, δv0, calibrationFrame, xData, mass, laserFreq, freqOffset, randomSampling=False):
  yData=[]; yerr=[]

  for run in calibrationFrame.index:
    fa = calibrationFrame.loc[run,'centroid']+freqOffset; δfa = calibrationFrame.loc[run,'cent_uncertainty']
    if randomSampling: fa = np.random.normal(loc=fa, scale=δfa) #random sample for centroid frequency based on fit statistics
    ΔEkin =calibrationFrame.loc[run,'ΔEkin']; yData+=[ΔEkin]
    δΔEkin=calibrationFrame.loc[run,'ΔEkin_uncertainty']; yerr+=[δΔEkin]
  res = LinearModel().fit(np.array(yData), x=xData, slope=1, intercept=0, weights=1/np.array(yerr), method='leastsq', fit_kws={'xtol': 1E-6, 'ftol':1E-6})
  return(res)

def uploadBeamEnergyCorrections(dates):
  beamEnergyCorrectionForDate={}
  for date in dates: beamEnergyCorrectionForDate[date]=float('NaN')
  for tString in ['P12-S12']:#,'P32-S12','P12-D32','P32-D32','P32-D52']:
    try:
      with open(tString+'/energyCorrectionDic.pkl','rb') as file: energyCorrectionFromFile=pickle.load(file); file.close()
    except: pass
    print('energyCorrectionFromFile:', energyCorrectionFromFile)
    #   energyCorrectionFromFile={}; print('wheee')
    #   for date in datesForTransition[tString]: energyCorrectionFromFile[date]=float('NaN')
    beamEnergyCorrectionForDate.update(energyCorrectionFromFile)
  for i in range(len(dates)):
    if np.isnan(beamEnergyCorrectionForDate[dates[i]]):
      j=0
      while np.isnan(beamEnergyCorrectionForDate[dates[i-j]]): j+=1
      beamEnergyCorrectionForDate[dates[i]] = beamEnergyCorrectionForDate[dates[i-j]] #for days without beam energy data, I'll just assume it's the same as most recent day with data.
  return(beamEnergyCorrectionForDate)

def updateBeamEnergyCorrections(transitionString, mass, beta, datesForTransition, colinearRunsForDate, anticolinearRunsForDate, laserDic, compCoParmResults, compAntiParmResults,δlaserFreq=1):
  beamEnergyCorrectionForDate={}
  v0_estimates=[]
  print('anticolinearRunsForDate:', anticolinearRunsForDate)
  for date in datesForTransition[transitionString]:
    energyCorrectionList=[]
    try:colinearRuns=np.array(colinearRunsForDate[date])
    except KeyError: colinearRunsForDate[date]=np.array([]); colinearRuns=colinearRunsForDate[date]
    try:anticolinearRuns=np.array(anticolinearRunsForDate[date])
    except KeyError:anticolinearRunsForDate[date]=np.array([]); anticolinearRuns=anticolinearRunsForDate[date]
    laserFreqsCo=np.array([laserDic[run] for run in colinearRuns]); uniqueLaserFreqsCo=np.unique(laserFreqsCo)
    laserFreqsAnti=[laserDic[run] for run in anticolinearRuns]; uniqueLaserFreqsAnti=np.unique(laserFreqsAnti)
    
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
        print('test?', date, lfc, lfa, ΔEkin)
        energyCorrectionList+=[ΔEkin]
        v0_estimates+=[bootstrapUncertainty(get_v0,2000,[ [mass,0],[lfc,δlaserFreq],[lfa,δlaserFreq],[fc,δfc],[fa,δfa] ])]

      # some shit for generating a bootstrapping convergence plot
      # convergenceArray=[]
      # nSamples = np.array(list(range(2,20)) + list(range(20,5000,10))); print(nSamples)
      # for n in nSamples:
      #   print(n)
      #   convergenceArray+=[bootstrapUncertainty(get_v0, n,[ [mass,0],[lfc,δlaserFreq],[lfa,δlaserFreq],[fc,δfc],[fa,δfa] ])[1]]

      # plt.plot(nSamples, convergenceArray)
      # plt.xlabel('num Samples'); plt.ylabel(r'$\sigma_{v_0}$'); plt.title('Convergence of centroid error estimate from boostrapping')
      # plt.show()


    beamEnergyCorrectionForDate[date]=np.mean(energyCorrectionList)

  v0_final = weightedStats(*np.array(v0_estimates).transpose())
  print("v0_estimatesList:\n",v0_estimates,"\nv0_final:",v0_final) 
    
  try: 
    with open(transitionString+'/energyCorrectionDic.pkl','rb') as file: energyCorrectionFromFile=pickle.load(file); file.close()
  except: energyCorrectionFromFile={}
  if beamEnergyCorrectionForDate==energyCorrectionFromFile: print("oohwee"); energyCorrectionDicChanged=False
  else:
    with open(transitionString+'/energyCorrectionDic.pkl','wb') as file: energyCorrectionFromFile=pickle.dump(beamEnergyCorrectionForDate, file); file.close(); energyCorrectionDicChanged=True
  return(v0_final)

def updateLaserDic(logDirectory):
  laserFreqDic={}
  
  logFileList=os.listdir(logDirectory)#["August_30_2022_Al.xlsx",]
  for logFile in logFileList:
    if '.xlsx' in logFile:
      print('importing log file ', logFile)
      logFrame=pd.read_excel(logDirectory+logFile,usecols=['Run #', 'Laser Freq. (THz)']).dropna()
      #logFile=np.array(logFile)
      for index, row in logFrame.iterrows():
        laserFreqDic[int(row['Run #'])]=row['Laser Freq. (THz)']*3E6
  #print(laserFreqDic)
  with open('laserDic.pkl','wb') as file: pickle.dump(laserFreqDic, file); file.close()

def weightedStats(values, errors):
  try: sigmaPropagated=1/np.sum(1/errors); weightArray=sigmaPropagated/errors; weightedMean=np.sum(weightArray*values)
  except: sigmaPropagated=float('NaN'); weightedMean=float('NaN')

  return(weightedMean, sigmaPropagated, np.std(values))

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

def propogateBeamEnergyCorrectionUncertainties(massTuple, vcTuple, fcTuple, v0Tuple):
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

def get_average_v0(mass, vc, va1, va2, fc, fa1, fa2):
  '''function just for this case where I have two anticolinear beam frequencies to calculate a beam energy correction from'''
  v0_1=calculateBeamEnergyCorrection(mass, vc, va1, fc, fa1)[1]
  v0_2=calculateBeamEnergyCorrection(mass, vc, va2, fc, fa2)[1]
  v0_avg = (v0_1+v0_2)/2
  return(v0_avg)

def approximateColinearFrequencyShift(vc, ΔEkin):
  beta2=np.sqrt(1-((mass*amu2eV)/(np.array(29915+ΔEkin)+mass*amu2eV))**2);
  approxColinearFrequencyShift=(np.sqrt((1-beta)/(1+beta))-np.sqrt((1-beta2)/(1+beta2)))*vc
  #print(approxColinearFrequencyShift)
  return(approxColinearFrequencyShift)

def spShiftPredictor(mass, ΔEinelastic, laserFreq, colinear=True): #predicting spShift frequency based on the the energy loss in eV due to inelastic scattering in CEC 
    nominalE=29915; amu2eV = np.float64(931494102.42)
    beta1=np.sqrt(1-((mass*amu2eV)/(nominalE+mass*amu2eV))**2)
    beta2=np.sqrt(1-((mass*amu2eV)/((nominalE+ΔEinelastic)+mass*amu2eV))**2);
    if colinear: approxFrequencyShift=-(np.sqrt((1-beta1)/(1+beta1))-np.sqrt((1-beta2)/(1+beta2)))*laserFreq
    else:        approxFrequencyShift=-(np.sqrt((1+beta1)/(1-beta1))-np.sqrt((1+beta2)/(1-beta2)))*laserFreq
    return(approxFrequencyShift)

def analyzeTransition(scanDirec, transitionString, mass, targetDirectoryName, datesForTransition,colinearRunsForDate, anticolinearRunsForDate, laserDic, redoFits=False, exportSpectrum=False, eCorrectionsForRun=False, ccg=0,acg=0,
  inelasticSidepeak=False, fixed_Alower=False, fixed_Aupper=False, freqOffset=0, tofWindow=[450,550],cuttingColumn='time_step', equal_fwhm=False, cec_sim_data_path=False):
  directoryPrefix='results'+'/equal_fwhm_'+str(equal_fwhm)+'/cec_sim_toggle_'+str(cec_sim_data_path!=False)#'spectralData'
  nuclearSpin=[2.5]
  massNumber=27
  jGround=0.5; jExcited=0.5
  peakModel='pseudoVoigt'
  colinearRuns=[]; anticolinearRuns=[]
  colinearFreqs=[]; anticolinearFreqs=[]
  for date in datesForTransition[transitionString]:
    try: colinearRuns+=colinearRunsForDate[date]
    except KeyError: colinearRunsForDate[date]=[]; colinearRuns += colinearRunsForDate[date]
    try: anticolinearRuns+=anticolinearRunsForDate[date]
    except KeyError: anticolinearRunsForDate[date]=[]; anticolinearRuns += anticolinearRunsForDate[date]

  compiledColinearParmResults=pd.DataFrame()
  compiledAnticolinearParmResults=pd.DataFrame()
  for run in colinearRuns:
    colinearity=True
    targetDirectory=targetDirectoryName+'/Colinear/Scan%d'%run
    eco=False if eCorrectionsForRun==False else eCorrectionsForRun[run]
    spectrumKwargs={'runs':[run], 'mass':massNumber, 'jGround':jGround, 'jExcited':jExcited, 'nuclearSpinList':nuclearSpin, 'laserFrequency':laserDic[run],
                    'colinearity':colinearity, 'directoryPrefix':directoryPrefix,'targetDirectory':targetDirectory, 'scanDirectory':scanDirec,
                    'windowToF':tofWindow,'cuttingColumn':cuttingColumn, 'constructSpectrum':False}
    fittingKwargs ={'colinearity':False, 'cec_sim_data_path':cec_sim_data_path,'equal_fwhm':equal_fwhm, 'peakModel':peakModel,'transitionLabel':'P12-S12'}
    spec=spc.Spectrum(**spectrumKwargs); spec.fitAndLogData(**fittingKwargs);
    popFrame=spec.populateFrame(prefix="iso0",index=run); compiledColinearParmResults=pd.concat([compiledColinearParmResults, popFrame])

  for run in anticolinearRuns:
    colinearity=False
    targetDirectory=targetDirectoryName+'/Anticolinear/Scan%d'%run
    eco=False if eCorrectionsForRun==False else eCorrectionsForRun[run]
    spectrumKwargs={'runs':[run], 'mass':massNumber, 'jGround':jGround, 'jExcited':jExcited, 'nuclearSpinList':nuclearSpin, 'laserFrequency':laserDic[run],
                    'colinearity':colinearity, 'directoryPrefix':directoryPrefix,'targetDirectory':targetDirectory, 'scanDirectory':scanDirec,
                    'windowToF':tofWindow,'cuttingColumn':cuttingColumn, 'constructSpectrum':True}#False}
    fittingKwargs ={'colinearity':False, 'cec_sim_data_path':cec_sim_data_path,'equal_fwhm':equal_fwhm, 'peakModel':peakModel,'transitionLabel':'P12-S12'}
    spec=spc.Spectrum(**spectrumKwargs); spec.fitAndLogData(**fittingKwargs);
    popFrame=spec.populateFrame(prefix="iso0",index=run); compiledAnticolinearParmResults=pd.concat([compiledAnticolinearParmResults, popFrame])

  return(compiledColinearParmResults, compiledAnticolinearParmResults)

def getEnergyCorrectedResults(scanDirec, targetDirectoryName, transitionString, mass, beta, datesForTransition, colinearRunsForDate, anticolinearRunsForDate, laserDic, colinearCentroidGuess, anticolinearCentroidGuess,redoFits=False,
                              exportSpectrum=False, inelasticSidepeak=False, fixed_Alower=False, fixed_Aupper=False, freqOffset=0, tofWindow=[450,550],cec_sim_data_path=False, equal_fwhm=False, redoFitWithEnergyCorrection=False):
  compiledColinearParmResults, compiledAnticolinearParmResults = analyzeTransition(scanDirec, transitionString, mass, targetDirectoryName, datesForTransition, colinearRunsForDate, anticolinearRunsForDate, laserDic, 
                                                                                   redoFits=redoFits, eCorrectionsForRun=False, ccg=colinearCentroidGuess[transitionString],acg=anticolinearCentroidGuess[transitionString],
                                                                                   inelasticSidepeak=inelasticSidepeak, fixed_Alower=fixed_Alower, fixed_Aupper=fixed_Aupper, freqOffset=freqOffset, tofWindow=tofWindow,
                                                                                   cec_sim_data_path=cec_sim_data_path, equal_fwhm=equal_fwhm, exportSpectrum=exportSpectrum)
  correctedCentroidEstimate = updateBeamEnergyCorrections(transitionString, mass, beta, datesForTransition, colinearRunsForDate, anticolinearRunsForDate, laserDic, compiledColinearParmResults, compiledAnticolinearParmResults)
  beamEnergyCorrectionForDate=uploadBeamEnergyCorrections(np.array(list(datesForTransition.values())).flatten())# TODO: rename this function
  energyCorrectionForRun={};
  for date in datesForTransition[transitionString]:
    for run in np.append(colinearRunsForDate[date], anticolinearRunsForDate[date]): energyCorrectionForRun[int(run)] = beamEnergyCorrectionForDate[date] #I'll use this providing energy corrections on a per-run basis when redoing my fits 
  if np.isnan(correctedCentroidEstimate[0]):
    r=colinearRunsForDate[datesForTransition[transitionString][-1]][-1];
    approximateCentroidShift=approximateColinearFrequencyShift(laserDic[r], energyCorrectionForRun[r])
    ccGuess=colinearCentroidGuess[transitionString]; acGuess=anticolinearCentroidGuess[transitionString]
    if ccGuess!=0: ccGuess-=approximateCentroidShift; acGuess=ccGuess
  else: ccGuess=acGuess=correctedCentroidEstimate[0]
  if redoFitWithEnergyCorrection:
    compiledColinearParmResults_Corrected, compiledAnticolinearParmResults_Corrected = analyzeTransition(scanDirec, transitionString, mass, targetDirectoryName, datesForTransition, colinearRunsForDate, anticolinearRunsForDate, laserDic,
                                                                                                         redoFits=redoFits, eCorrectionsForRun=energyCorrectionForRun, ccg=colinearCentroidGuess[transitionString],acg=anticolinearCentroidGuess[transitionString],
                                                                                                         inelasticSidepeak=inelasticSidepeak, fixed_Alower=fixed_Alower, fixed_Aupper=fixed_Aupper, freqOffset=freqOffset, tofWindow=tofWindow,
                                                                                                         cec_sim_data_path=cec_sim_data_path, equal_fwhm=equal_fwhm, exportSpectrum=True)
  
    return(correctedCentroidEstimate, compiledColinearParmResults, compiledAnticolinearParmResults,compiledColinearParmResults_Corrected, compiledAnticolinearParmResults_Corrected)
  else:
    return(correctedCentroidEstimate, compiledColinearParmResults, compiledAnticolinearParmResults, -1, -1)
        

def main(cec_sim_data_path=False, equal_fwhm=False, redoFitWithEnergyCorrection=False, redoFits=False):

  tofWindow=[485,550]
  freqOffset=1129900000
  logDirec="BeamEnergyCalibrationData\\"
  # updateLaserDic(logDirec)
  scanDirec='BeamEnergyCalibrationData'
  
  targetDirectoryName='beamEnergy_analysis'
  mass=26.98153841; amu2eV = np.float64(931494102.42); beta=np.sqrt(1-((mass*amu2eV)/(np.array(29915)+mass*amu2eV))**2) #nominal beta value for approximating f0, used for frequency offset applied to datasets prior to fitting 
  iNuc27=sympy.Rational(5,2); jElecP12=sympy.Rational(1,2);jElecS12=sympy.Rational(1,2);
  jLower={}; jUpper={}
  jLower['P12-S12']=jElecP12; jUpper['P12-S12']=jElecS12;


  colinearCentroidGuess=0; anticolinearCentroidGuess=0
  #colinearRuns={}; antiColinearRuns={}; laserFreqCo={}; laserFreqAnti={}
  newLogFiles=False
  if newLogFiles: updateLaserDic(logDirec)
  with open('laserDic.pkl','rb') as file: laserDic=pickle.load(file); file.close()
  #print("LASERDIC:\n",laserDic)

  #plt.plot(laserDic.keys(), laserDic.values(),'b.')
  #print('unique laser frequencies:',np.unique(list(laserDic.values())))
  #plt.show()

  #dates=['02May2024']
  datesForTransition={};
  datesForTransition['P12-S12']=['02May2024']


  colinearCentroidGuess={}; anticolinearCentroidGuess={}
  colinearCentroidGuess['P12-S12'] = 0 ; anticolinearCentroidGuess['P12-S12'] = 0

  colinearRunsForDate={}; anticolinearRunsForDate={}
  #'P12-S12':
  anticolinearRunsForDate['02May2024'] = [16253,16254,16255,16263,16264,16265]  #16273?
  colinearRunsForDate['02May2024']     = [16258,16259,16260,16268,16269,16270]

  ΔEinelastic=False#np.mean([4.079489344597626, 4.181311602875548]) #in eV, as measured from runs 14470 and 14471 on Nov 8 2022 (see InvestigatingSidePeaks_voltageSpace.py)
  
  #compiling all the beam-energy corrected datasets
  

  v0_final, P12_S12_Colinear, P12_S12_Anticolinear, P12_S12_Colinear_Corrected, P12_S12_Anticolinear_Corrected = getEnergyCorrectedResults(scanDirec, targetDirectoryName, 'P12-S12', mass, beta, datesForTransition, colinearRunsForDate, anticolinearRunsForDate, laserDic, colinearCentroidGuess, anticolinearCentroidGuess,redoFits=redoFits,
    inelasticSidepeak=ΔEinelastic, fixed_Alower=False, fixed_Aupper=False, freqOffset=freqOffset, tofWindow=tofWindow, cec_sim_data_path=cec_sim_data_path, equal_fwhm=equal_fwhm, redoFitWithEnergyCorrection=redoFitWithEnergyCorrection);

  # colorList=['r','b','orange','purple']
  # for i, dic in enumerate([P12_S12_Colinear, P12_S12_Anticolinear, P12_S12_Colinear_Corrected, P12_S12_Anticolinear_Corrected]):
  #   for key in list(dic.keys()):
  #     plt.errorbar(x=int(key), y=dic[key]['iso0_centroid'].value-freqOffset,yerr=dic[key]['iso0_centroid'].stderr, fmt='.', color=colorList[i])
  # plt.ylabel('centroid (MHz) - %.2fTHz'%(freqOffset/1E6));  plt.xlabel('run')

  # from matplotlib.lines import Line2D
  # from matplotlib.ticker import MaxNLocator
  # legend_elements = [Line2D([0], [0], marker='o', color='w', label='colinear', markerfacecolor='red', markersize=10),
  #                    Line2D([0], [0], marker='o', color='w', label='anticolinear', markerfacecolor='blue', markersize=10),
  #                    Line2D([0], [0], marker='o', color='w', label='col, corrected', markerfacecolor='orange', markersize=10),
  #                    Line2D([0], [0], marker='o', color='w', label='anti, corrected', markerfacecolor='purple', markersize=10),]
  # plt.legend(handles=legend_elements, loc = (0.01,0.15))
  # plt.title("02May2024 Beam Energy Analysis")
  # ax = plt.gcf().gca().xaxis.set_major_locator(MaxNLocator(integer=True))
  # plt.show()

  return(v0_final)

if __name__==  '__main__': 
  cec_sim_toggle_list = [False]#["27Al_CEC_peaks.csv", False]
  equal_fwhm_toggle_list = [True]#[False, True]
  for cec_sim_toggle in cec_sim_toggle_list:
    for equal_fwhm_toggle in equal_fwhm_toggle_list:
      main(cec_sim_data_path=cec_sim_toggle, equal_fwhm=equal_fwhm_toggle)

#TODO: pass cec_sim_toggle and fwhm_toggle through BEA calls in onlineAnalysis.py