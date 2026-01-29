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
runsDictionary = {
  27:[16368,16369,16370,16389,16391,16392,16395,16396,16397,16410,16412,16413,16422,16424,16425,16426,#16367, 16371,16372,16373,16374,16375,16376 are all trash #16428 not good #16366 a little weird
      16429,16430,16439,16441,16442,16451,16458,16459,16470,16473,16474,16477,16492,16494,16495,16508,16510,16512]}
jGround=0.5
jExcited=0.5
iNucDictionary={27:[2.5]}
massDictionary = {27:26.981538408}
mass_uncertaintyDictionary = {27:0.00000005}
laserDictionary = {27:376.052850}
timeStepDictionary= {27:[489,543]}
tofDictionary={27: [23.45E-6,26.0E-6]}

class WhatToRun: #Class for passing arguments that determine whether to run spectrum construction and fits from scratch or not
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
    # with open(exportsPrefix+'calibrationVsScanNumber_fit_report.txt','w') as file: file.write(calibrationVsScanNumber.fit_report()); file.close()
    # with open(exportsPrefix+'calibrationVsScanTime_fit_report.txt','w') as file: file.write(calibrationVsScanTime.fit_report()); file.close()
    with open(exportsPrefix+'calibrationRelevantConstants.txt','w') as file: file.write('scanTimeOffset: %d\nv0: '%scanTimeOffset +str(v0) );file.close()   
  return(calibrationFrame, calibrationVsScanNumber, calibrationVsScanTime)

def commissAnalysis(equal_fwhm = False, cec_sim_toggle = "27Al_CEC_peaks.csv", peakModel='pseudoVoigt', whatToRun=False):
  directoryPrefix='results'+'/equal_fwhm_'+str(equal_fwhm)+'/cec_sim_toggle_'+str(cec_sim_toggle!=False)
  if not os.path.exists(directoryPrefix):  os.makedirs(directoryPrefix)
  wtr = whatToRun if whatToRun else WhatToRun() #determines whether to run spectrum construction and fits from scratch or not.

  beamEnergyAnalysisResults = bea.main(equal_fwhm = equal_fwhm, cec_sim_data_path = cec_sim_toggle, redoFits=wtr.fitAndLogToggle_BEA, redoFitWithEnergyCorrection=False);
  print(beamEnergyAnalysisResults)
  v0 = beamEnergyAnalysisResults[0]; δv0 = np.sqrt(beamEnergyAnalysisResults[1]**2+beamEnergyAnalysisResults[2]**2)
  print(f'v0={v0}+/-{δv0}')
  massNumber=27
  colinearity = False
  spectrumKwargs={'mass':massDictionary[massNumber],'mass_uncertainty':mass_uncertaintyDictionary[massNumber], 'jGround':jGround, 'jExcited':jExcited, 'nuclearSpinList':iNucDictionary[massNumber],
                  'laserFrequency':3E6*laserDictionary[massNumber],'colinearity':colinearity, 'directoryPrefix':directoryPrefix,'scanDirectory':str(massNumber)+'Al/',
                  'timeOffset':scanTimeOffset,'windowToF':tofDictionary[massNumber], 'cuttingColumn':'ToF', 'constructSpectrum':wtr.exportSpectrumToggle_calibration}
  fittingKwargs ={'colinearity':False, 'cec_sim_data_path':cec_sim_toggle,'equal_fwhm':equal_fwhm, 'peakModel':peakModel,'transitionLabel':'P12-S12'}
  
  calibrationFrame_beforeBEC, calibrationVsScanNumber, calibrationVsScanTime = calibrationProcedure(runsDictionary[27],v0,δv0, spectrumKwargs=spectrumKwargs, fittingKwargs=fittingKwargs)
  calibrationFrame_beforeBEC.to_csv(f'{directoryPrefix}/CalibrationDiagnostics/calibrationFrame_beforeBEC.csv')

  #Applying best fit linear energy correction to each scan, and then repeating fitting and exporting procedures
  spectrumKwargs['energyCorrection']=calibrationVsScanTime; spectrumKwargs['constructSpectrum']=wtr.exportSpectrumToggle_calibration_bec
  calibrationFrame, _, _ = calibrationProcedure(runsDictionary[27],v0,δv0, spectrumKwargs=spectrumKwargs, fittingKwargs=fittingKwargs)
  calibrationFrame.to_csv(f'{directoryPrefix}/CalibrationDiagnostics/calibrationFrame_afterBEC.csv')
  return(calibrationFrame)
  
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
      calibrationFrame=commissAnalysis(equal_fwhm = equal_fwhm_toggle, cec_sim_toggle = cec_sim_toggle, peakModel=peakModel, whatToRun=wtr)
      i+=1; print('i=',i)
      print(calibrationFrame)