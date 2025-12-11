import numpy as np
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
import hyperfinePredictorGREAT as hpg
import spectrumHandler as sh
import BeamEnergyAnalysis as bea
import SpectrumClass as spc

scanTimeOffset=1716156655
# freqOffset=1129900000
runsDictionary = {
  27:[16368,16369,16370,16389,16391,16392,16395,16396,16397,16410,16412,16413,16422,16424,16425,16426,#16367, 16371,16372,16373,16374,16375,16376 are all trash #16428 not good #16366 a little weird
      16429,16430,16439,16441,16442,16451,16458,16459,16470,16473,16474,16477,16492,16494,16495,16508,16510,16512]}
jGround=0.5
jExcited=0.5
iNucDictionary={27:[2.5]}
massDictionary = {27:26.981538408}
mass_uncertaintyDictionary = {27:0.00000005}
laserDictionary = {27:376.052850}
timeStepDictionary= {27:[489,543]}#[485,550]}
tofDictionary={27: [23.45E-6,26.0E-6]}

class WhatToRun:#TODO:
  def __init__(self):
    self.fitAndLogToggle_BEA =                True;#False;#
    self.exportSpectrumToggle_calibration =   True;#False;#
    self.fitAndLogToggle_calibration =        True;#False;#
    self.exportSpectrumToggle_calibration_bec=True;#False;#
    self.fitAndLogToggle_calibration_bec=     True;#False;#
    self.exportSpectrumToggle     =           True;#False;#
    self.exportSpectrumToggle_bec =           True;#False;#
    self.fitAndLogToggleDic={  27:            True}#False}#


def calibrationProcedure(calibrationScans, v0, δv0, spectrumKwargs={}, fittingKwargs={}):
  calibrationFrame = pd.DataFrame()
  mass=spectrumKwargs['mass']
  laserFreq=spectrumKwargs['laserFrequency']
  for run in calibrationScans:
    print('run%d'%run)
    spec=spc.Spectrum(runs=[run], targetDirectory=f'Scan{run}', **spectrumKwargs)           
    spec.fitAndLogData(**fittingKwargs); popFrame=spec.populateFrame(prefix="iso0",index=run)
    fa= spec.resultParams['iso0_centroid'].value; δfa= spec.resultParams['iso0_centroid'].stderr
    ΔEkin =bea.calculateBeamEnergyCorrectionFromv0vc(mass, laserFreq, fa, v0)
    δΔEkin=bea.propagateBeamEnergyCorrectionUncertainties([mass,0], [laserFreq,1], [fa, δfa], [v0,δv0])
    popFrame['ΔEkin']=ΔEkin; popFrame['ΔEkin_uncertainty']=δΔEkin
    calibrationFrame=pd.concat([calibrationFrame, popFrame])
  calibrationScanTimes=np.array(calibrationFrame['avgScanTime'])
  calibrationVsScanNumber = bea.getCalibrationFunction(v0, δv0, calibrationFrame, np.array(calibrationFrame.index),mass, laserFreq)
  calibrationVsScanTime   = bea.getCalibrationFunction(v0, δv0, calibrationFrame, calibrationScanTimes, mass, laserFreq)
  directoryPrefix=spectrumKwargs['directoryPrefix']
  energyCorrected=spectrumKwargs['energyCorrection'] if 'energyCorrection' in spectrumKwargs.keys() else False
  '''exporting calibration results for analysis comparision purposes'''
  exportsPrefix='./'+directoryPrefix+'/CalibrationDiagnostics/'
  if energyCorrected==False:
    if not os.path.isdir(exportsPrefix): os.makedirs(exportsPrefix)
    with open(exportsPrefix+'calibrationVsScanNumber_fit_report.txt','w') as file: file.write(calibrationVsScanNumber.fit_report()); file.close()
    with open(exportsPrefix+'calibrationVsScanTime_fit_report.txt','w') as file: file.write(calibrationVsScanTime.fit_report()); file.close()
    with open(exportsPrefix+'calibrationRelevantConstants.txt','w') as file: file.write('scanTimeOffset: %d\nv0: '%scanTimeOffset +str(v0) );file.close()
    calibrationFrame[['centroid','cent_uncertainty', 'ΔEkin','ΔEkin_uncertainty','avgScanTime']].to_csv(exportsPrefix+'calibrationFunctionData.csv')
    plt.title('calibration run beam energy corrections'); plt.xlabel('run'); plt.ylabel('energy corrections (eV)')
    plt.errorbar(calibrationFrame.index, y=calibrationFrame['ΔEkin'], yerr=calibrationFrame['ΔEkin_uncertainty'], fmt='k.', label='individual runs')
    #plt.gca().axhline(y=allIsotopesFrame.loc[stableIndex]['centroid'], label='all runs combined')~
    plt.plot(calibrationFrame.index, calibrationVsScanNumber.best_fit, 'b-', label = 'linear correction as function of run number')
    plt.legend(loc=2)
    plt.savefig(exportsPrefix+'/energy_correctionsVsRunNumber.png');plt.close()

    plt.title('calibration run beam energy corrections'); plt.xlabel('time (s)'); plt.ylabel('energy corrections (eV)')
    plt.errorbar(calibrationFrame['avgScanTime'], y=calibrationFrame['ΔEkin'], yerr=calibrationFrame['ΔEkin_uncertainty'], fmt='k.', label='individual runs')
    #plt.gca().axhline(y=allIsotopesFrame.loc[stableIndex]['centroid'], label='all runs combined')
    plt.plot(calibrationFrame['avgScanTime'], calibrationVsScanTime.best_fit, 'b-', label = 'linear correction as function of time')
    plt.legend(loc=2)
    plt.savefig(exportsPrefix+'/energy_correctionsVsRunTime.png');plt.close()
  else:
    plt.title('calibration run A Ratios'); plt.xlabel('run'); plt.ylabel(r'$A_{Lower}/A_{Upper}$')
    plt.errorbar(calibrationFrame.index, y=calibrationFrame['aRatio'], yerr=calibrationFrame['aRatio_uncertainty'], fmt='k.', label='individual runs')
    plt.gca().axhline(y=np.mean(calibrationFrame['aRatio']), label='avg A Ratio')
    plt.legend(loc='best')
    plt.savefig(exportsPrefix+'/aRatioVsRun_energyCorrected.png');plt.close()

    plt.title('calibration run centroids'); plt.xlabel('run'); plt.ylabel('centroid (MHz)')
    plt.errorbar(calibrationFrame.index, y=calibrationFrame['centroid'], yerr=calibrationFrame['cent_uncertainty'], fmt='k.', label='individual runs')
    plt.gca().axhline(y=np.mean(calibrationFrame['centroid']), label='avg centroid')
    plt.legend(loc='best')
    plt.savefig(exportsPrefix+'/centroidVsRun_energyCorrected.png');plt.close()
  return(calibrationFrame, calibrationVsScanNumber, calibrationVsScanTime)

def fullAnalysis(a_ratio_fixed = True, equal_fwhm = False, cec_sim_toggle = "27Al_CEC_peaks.csv", spinList22=[4], peakModel='pseudoVoigt',whatToRun=False):
  directoryPrefix='results'+'/equal_fwhm_'+str(equal_fwhm)+'/cec_sim_toggle_'+str(cec_sim_toggle!=False)
  if not os.path.exists(directoryPrefix):  os.makedirs(directoryPrefix)
  if whatToRun==False: wtr=WhatToRun()#default to settings where everything runs.
  else: wtr=whatToRun

  beamEnergyAnalysisResults = bea.main(equal_fwhm = equal_fwhm, cec_sim_data_path = cec_sim_toggle, redoFits=wtr.fitAndLogToggle_BEA, redoFitWithEnergyCorrection=False); #print(beamEnergyAnalysisResults)
  v0 = beamEnergyAnalysisResults[0]; δv0 = np.sqrt(beamEnergyAnalysisResults[1]**2+beamEnergyAnalysisResults[2]**2)
  print(f'v0={v0}+/-{δv0}')
  colinearity = False
  
  '''starting with calibration runs'''
  massNumber=27
  
  spectrumKwargs={'mass':massDictionary[massNumber],'mass_uncertainty':mass_uncertaintyDictionary[massNumber], 'jGround':jGround, 'jExcited':jExcited, 'nuclearSpinList':iNucDictionary[massNumber],
                    'laserFrequency':3E6*laserDictionary[massNumber],'colinearity':colinearity, 'directoryPrefix':directoryPrefix,'scanDirectory':str(massNumber)+'Al/',
                    'timeOffset':scanTimeOffset,'windowToF':tofDictionary[massNumber], 'cuttingColumn':'ToF', 'constructSpectrum':wtr.exportSpectrumToggle_calibration}
  fittingKwargs ={'colinearity':False, 'cec_sim_data_path':cec_sim_toggle,'equal_fwhm':equal_fwhm, 'peakModel':peakModel,'transitionLabel':'P12-S12'}
  
  calibrationFrame_beforeBEC, calibrationVsScanNumber, calibrationVsScanTime = calibrationProcedure(runsDictionary[27],v0,δv0, spectrumKwargs=spectrumKwargs, fittingKwargs=fittingKwargs)
  spectrumKwargs['energyCorrection']=calibrationVsScanTime; spectrumKwargs['constructSpectrum']=wtr.exportSpectrumToggle_calibration_bec
  calibrationFrame, _, _ = calibrationProcedure(runsDictionary[27],v0,δv0, spectrumKwargs=spectrumKwargs, fittingKwargs=fittingKwargs)
  calibrationFrame_beforeBEC.to_csv(f'{directoryPrefix}/CalibrationDiagnostics/calibrationFrame_beforeBEC.csv')
  calibrationFrame.to_csv(f'{directoryPrefix}/CalibrationDiagnostics/calibrationFrame_afterBEC.csv')
  #print(calibrationFrame)

    
  def beamEnergyBootstrapping(v0, δv0, calibrationFrame, xEval, mass, laserFreq, numTrials):
    yEvals = np.zeros((len(xEval),numTrials))
    for i in range(numTrials):
      v_sample = np.random.normal(loc=v0, scale=δv0)
      res=bea.getCalibrationFunction(v_sample, δv0, calibrationFrame, xEval, mass, laserFreq, randomSampling=True)
      yEvals[:,i] = res.eval(x=xEval)
    return(yEvals)

  '''section for estimating error in beam energy corrections'''
  # isoTimes=np.array(allIsotopesFrame['avgScanTime'])
  calibrationScanTimes=np.array(calibrationFrame['avgScanTime'])
  # t0=time.time()
  yEvals = beamEnergyBootstrapping(v0, δv0, calibrationFrame, calibrationScanTimes, massDictionary[27], 3E6*laserDictionary[27], 200)
  # t1=time.time()
  print(yEvals)
  #print('time elapsed',t1-t0)
  # quit()

  # for i in range(len(isoTimes)):
  #   plt.plot(yEvals[i,:], label=allIsotopesFrame.loc[i,'massNumber']);
  # plt.xlabel('trial i'); plt.ylabel('beam energy corrections from calibration fit for bootstrap trial i'); plt.title('Random Sampling Beam Energy Corrections')
  # plt.legend()
  # plt.show()

  # convergenceArray=[]
  # nSamples = np.array(list(range(2,20)) + list(range(20,250,5)));
  # for n in nSamples:
  #   print('n=%d'%n)
  #   yEvals = beamEnergyBootstrapping(v0, δv0, calibrationFrame, calibrationScanTimes, massDictionary[27], 3E6*laserDictionary[27], n)
  #   convergenceArray+=[np.std(yEvals)]

  # plt.plot(nSamples, convergenceArray)
  # plt.xlabel('num Samples'); plt.ylabel(r'Beam energy correction $(\langle t\rangle_{\text{all scans}})$'); plt.title('Convergence of error estimate for beam energy corrections')
  # plt.show()

  # convergenceArray=[[] for i in allIsotopesFrame.index]
  # nSamples = np.array(list(range(2,20)) + list(range(20,250,5)));

  # for i in allIsotopesFrame.index:
  #   plt.plot(nSamples, convergenceArray[i], label=isotopeShiftsSamples[i][0])
  # plt.xlabel('num Samples'); plt.ylabel(r'Beam energy corrected  $δv_{i,27}$(MHz)'); plt.title('Convergence of error estimate for isotope shifts from beam energy correction')
  # plt.legend(loc=1)
  # plt.show()

  xData =  np.array(calibrationFrame['avgScanTime']).astype(float) ; yData = np.array(calibrationFrame_beforeBEC['centroid']).astype(float)
  plt.plot(xData, yData)
  plt.show()

if __name__ == '__main__':
  wtr=WhatToRun()
  wtr.fitAndLogToggle_BEA =                False;#True;#
  wtr.exportSpectrumToggle_calibration =   False;#True;#
  wtr.fitAndLogToggle_calibration =        False,#False;
  wtr.exportSpectrumToggle_calibration_bec=False;#True;#
  wtr.fitAndLogToggle_calibration_bec=     False,#False;
  wtr.fitAndLogToggleDic={  27:            False}#True}#
  peakModel='pseudoVoigt'
  equal_fwhm_toggle_list = [True]#,False]
  cec_sim_toggle_list = [False]#, "27Al_CEC_peaks.csv"]
  i=0
  allFramesDic={}
  for equal_fwhm_toggle in equal_fwhm_toggle_list:
    for cec_sim_toggle in cec_sim_toggle_list:
      print(equal_fwhm_toggle,cec_sim_toggle)
      allIsotopesFrame=fullAnalysis(equal_fwhm = equal_fwhm_toggle, cec_sim_toggle = cec_sim_toggle, spinList22=[4], peakModel=peakModel, whatToRun=wtr)
      allFramesDic[('fwhm_'+str(equal_fwhm_toggle),'cec_'+str(cec_sim_toggle))]=allIsotopesFrame
      i+=1
      print('i=',i)
  print(allIsotopesFrame)

def refCentroidTester(**kwargs):
  δlaserFreq=1
  v0_estimates=[]
  laserDic={}
  with open('laserDic.pkl','rb') as file: laserDic = pickle.load(file); file.close()
  '''temporary struggle code until I can refactor'''
  anticolinearRuns = [16253,16254,16255,16263,16264,16265]
  colinearRuns     = [16258,16259,16260,16268,16269,16270]
  colinearCentroids={}; anticolinearCentroids={}
  mass=oa.massDictionary[27]; targetDirectoryName='/beamEnergy_analysis'
  directoryPrefix='results'+'/equal_fwhm_'+str(equal_fwhm_toggle)+'/cec_sim_toggle_'+str(cec_sim_toggle!=False)
  for run in colinearRuns:
    targetDirectory=targetDirectoryName+'/Colinear/Scan%d'%run
    spectrumFrame = sh.loadSpectrumFrame(mass, targetDirectory, directoryPrefix=directoryPrefix)
    xData = np.array(spectrumFrame['dcf']);
    yData = np.array(spectrumFrame['countrate']); yUncertainty = np.array(spectrumFrame['uncertainty'])
    result=hpg.fitData(xData, yData, yUncertainty, mass, [5/2], .5,.5, transitionLabel='P12-S12', colinearity=True,**kwargs)
    colinearCentroids[run] = {'value':result.params['iso0_centroid'].value, 'stderr':result.params['iso0_centroid'].stderr}
  for run in anticolinearRuns:
    targetDirectory=targetDirectoryName+'/Anticolinear/Scan%d'%run
    spectrumFrame = sh.loadSpectrumFrame(mass, targetDirectory, directoryPrefix=directoryPrefix)
    xData = np.array(spectrumFrame['dcf']);
    yData = np.array(spectrumFrame['countrate']); yUncertainty = np.array(spectrumFrame['uncertainty'])
    result=hpg.fitData(xData, yData, yUncertainty, mass, [5/2], .5,.5, transitionLabel='P12-S12', colinearity=False,**kwargs)
    anticolinearCentroids[run] = {'value':result.params['iso0_centroid'].value, 'stderr':result.params['iso0_centroid'].stderr}

  laserFreqsCo=np.array([laserDic[run] for run in colinearRuns]); uniqueLaserFreqsCo=np.unique(laserFreqsCo)
  laserFreqsAnti=[laserDic[run] for run in anticolinearRuns]; uniqueLaserFreqsAnti=np.unique(laserFreqsAnti)
  for lfc in uniqueLaserFreqsCo:
    #print('colinearRuns: ',colinearRuns)
    colinearCentroidResults = [[colinearCentroids[run]['value'], colinearCentroids[run]['stderr']] for run in colinearRuns if laserDic[run]==lfc]
    colinearWeightedStats=bea.weightedStats(*np.array(colinearCentroidResults).transpose())
    fc=colinearWeightedStats[0]; #print('lfc:',lfc,'colinearWeightedStats:',colinearWeightedStats)
    δfc = np.sqrt(colinearWeightedStats[1]**2+colinearWeightedStats[2]**2)
    for lfa in uniqueLaserFreqsAnti:
      #print('anticolinearRuns: ',antiRuns)
      anticolinearCentroidResults = [[anticolinearCentroids[run]['value'], anticolinearCentroids[run]['stderr']] for run in anticolinearRuns if laserDic[run]==lfa]
      anticolinearWeightedStats=bea.weightedStats(*np.array(anticolinearCentroidResults).transpose())
      fa=anticolinearWeightedStats[0];                                                                            
      δfa = np.sqrt(anticolinearWeightedStats[1]**2+anticolinearWeightedStats[2]**2)
      ΔEkin, centroidEstimate=bea.calculateBeamEnergyCorrection(mass, lfc,lfa,fc, fa)
      # print('test?', lfc, lfa, ΔEkin, centroidEstimate)
      v0_estimates+=[bea.bootstrapUncertainty(bea.get_v0,2000,[ [mass,0],[lfc,δlaserFreq],[lfa,δlaserFreq],[fc,δfc],[fa,δfa] ])]
    # print(centroidEstimate)
    v0_final, v0_error1,v0_error2 = bea.weightedStats(*np.array(v0_estimates).transpose())
    v0_error = np.sqrt(v0_error1**2+v0_error2**2)
  return(v0_final, v0_error)