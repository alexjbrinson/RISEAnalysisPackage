import numpy as np
import pandas as pd
import time
import pickle
from hyperfinePredictorGREAT import * #_23Oct2024Backup
import spectrumHandler as sh
import BeamEnergyAnalysis as bea
#TODO scan time offset should be fed into exportSpectrum function, along with calibration model

scanTimeOffset=1716156655
freqOffset=1129900000
runsDictionary = {
  22:[16463,16464,16465,16478,16479,16480,16481,16482,16483,16484,16485,16486,16487,16488,16489,16490,16491,16497,16498,16499,16500,16501,16502,16503,16504,16505],#16464
  23:[16405,16406,16407,16408,16414,16415,16416,16418,16419,16420,16421], #16404
  24:[16434,16435,16436,16437,16438,16445,16446,16447,16448,16449,16450], #16445,16446 have different buncher settings
  25:[16384,16385,16386,16387,16388],
  27:[16368,16369,16370,16389,16391,16392,16395,16396,16397,16410,16412,16413,16422,16424,16425,16426,#16367, 16371,16372,16373,16374,16375,16376 are all trash #16428 not good #16366 a little weird
      16429,16430,16439,16441,16442,16451,16458,16459,16470,16473,16474,16477,16492,16494,16495,16508,16510,16512]}
jGround=0.5
jExcited=0.5
iNucDictionary={
  22:[4], #spin assignment is tentative
  23:[2.5],
  24:[4,1], #isomer is spin 1
  25:[2.5],
  26:[5,0], #isomer is spin 0
  27:[2.5]}
massDictionary = {
  22:22.01942311,#previous mass value: 22.01954000,# 
  23:23.00724440,
  24:23.99994760, #isomer excited by 425.81 (10) keV
  25:24.99042831,
  #26:25.98689188,
  27:26.981538408}

massUncertDictionary = {
  22:0.00000030,# aka .3keV old uncertainty: 400keV = 0.0004 amu,
  23:0.00000040,
  24:0.00000024, #isomer excitation uncertainty = .1 keV
  25:0.00000007,
  #26:,
  27:0.00000005}

laserDictionary = {
  22:375.990796,
  23:376.004732,
  24:376.017863,
  25:376.030178,
  27:376.052850}
  
timeStepDictionary= {
  22:[450,530],
  23:[450,530],
  24:[450,530],
  25:[450,530],
  26:[450,530],
  27:[489,543]}#[485,550]}

tofDictionary={
  22: [21.50E-6,23.5E-6],
  23: [21.80E-6,24.0E-6],
  24: [22.00E-6,24.5E-6],
  25: [22.75E-6,25.0E-6],
  27: [23.45E-6,26.0E-6]}

# voltageEstimateDic={
#   22: freqToVoltage(massDictionary[24], 3E6*laserDictionary[24], -150.644014, freqOffset=freqOffset),
#   24: freqToVoltage(massDictionary[22], 3E6*laserDictionary[22], -160.097928, freqOffset=freqOffset)
# }
nominalVoltage=29893.0

class WhatToRun:#TODO:
  def __init__(self):
    self.fitAndLogToggle_BEA =                True;#False;#
    self.exportSpectrumToggle_calibration =   True;#False;#
    self.fitAndLogToggle_calibration =        True;#False;#
    self.exportSpectrumToggle_calibration_bec=True;#False;#
    self.fitAndLogToggle_calibration_bec=     True;#False;#
    self.exportSpectrumToggle     =           True;#False;#
    self.exportSpectrumToggle_bec =           True;#False;#
    self.fitAndLogToggleDic={  22:            True,#False,#
                               23:            True,#False,#
                               24:            True,#False,#
                               25:            True,#False,#
                               27:            True}#False}#


def calibrationProcedure(calibrationScans, v0, δv0, equal_fwhm = False, cec_sim_toggle = False, tofWindow=[489,543], cuttingColumn='time_step', keepLessIntegratedBins=True,
  fixed_Aratio=False, fixed_spShift=False, fixed_spProp=False, spScaleable=False,
  energyCorrection=False, exportSpectrumToggle=False, fitAndLogToggle=False, freqOffset=0,scanTimeOffset=1716156655,directoryPrefix='spectralData', peakModel='pseudoVoigt'):
  if not os.path.exists(directoryPrefix):
    os.makedirs(directoryPrefix)
  calibrationFrame = pd.DataFrame(index=calibrationScans, columns=['aLower','aLower_uncertainty','aUpper','aUpper_uncertainty','aRatio','aRatio_uncertainty',
                                                                   'centroid','cent_uncertainty', 'sigma', 'sigma_uncertainty', 'ΔEkin', 'ΔEkin_uncertainty', 'avgScanTime'])#,'redchi'])
  massNumber=27
  scanDirec=str(massNumber)+'Al/'
  laserFreq=3E6*376.052850
  mass=26.98153841
  jGround=0.5; jExcited=0.5; iNuc=[2.5]
  colinearity=False

  kwargs={
    'colinearity':colinearity,'exportSpectrum':exportSpectrumToggle, 'fitAndLog':fitAndLogToggle, 'freqOffset':freqOffset, 'timeOffset':scanTimeOffset, 'tofWindow':tofWindow, 'cuttingColumn':cuttingColumn,
    'fixed_Aratio':fixed_Aratio, 'cec_sim_data_path':cec_sim_toggle, 'equal_fwhm':equal_fwhm, 'directoryPrefix':directoryPrefix, 'peakModel':peakModel, 'energyCorrection':energyCorrection, 'spScaleable':spScaleable}

  
  for run in calibrationScans:
    targetDirectoryName = 'Scan%d'%run
    print('run%d'%run)
    if (exportSpectrumToggle or fitAndLogToggle): pass
    processData(scanDirec, [run], laserFreq, mass, targetDirectoryName, iNuc, jGround, jExcited, **kwargs)
    result = loadFitResults(directoryPrefix, massNumber, targetDirectoryName, energyCorrection=energyCorrection)
    aLower=result['iso0_Alower'].value; aLower_uncertainty=result['iso0_Alower'].stderr
    aUpper=result['iso0_Aupper'].value; aUpper_uncertainty=result['iso0_Aupper'].stderr
    aRatio=aLower/aUpper; aRatio_uncertainty = (aRatio**2) *( (aLower_uncertainty/aLower)**2 + (aUpper_uncertainty/aUpper)**2)
    
    calibrationFrame.loc[run,'centroid']=result['iso0_centroid'].value;
    calibrationFrame.loc[run,'cent_uncertainty']=result['iso0_centroid'].stderr
    calibrationFrame.loc[run,'aLower']=aLower; calibrationFrame.loc[run,'aLower_uncertainty']=aLower_uncertainty
    calibrationFrame.loc[run,'aUpper']=aUpper; calibrationFrame.loc[run,'aUpper_uncertainty']=aUpper_uncertainty
    calibrationFrame.loc[run,'aRatio']=aRatio; calibrationFrame.loc[run,'aRatio_uncertainty']=aRatio_uncertainty
    calibrationFrame.loc[run,'sigma'] =result['iso0_sigma'].value; calibrationFrame.loc[run,'sigma_uncertainty']=result['iso0_sigma'].stderr
    calibrationFrame.loc[run,'gamma'] =result['iso0_gamma'].value; calibrationFrame.loc[run,'gamma_uncertainty']=result['iso0_gamma'].stderr
    #calibrationFrame.loc[run,'redchi'] =result['redchi'].value
    fa= result['iso0_centroid'].value+freqOffset; δfa= result['iso0_centroid'].stderr
    #print('centroid=',fa)
    ΔEkin =bea.calculateBeamEnergyCorrectionFromv0vc(mass, laserFreq, fa, v0); #print('test...', 'energyCorrection = ', energyCorrection, 'ΔEkin=',ΔEkin)
    δΔEkin=bea.propogateBeamEnergyCorrectionUncertainties([mass,0], [laserFreq,1], [fa, δfa], [v0,δv0]); #print('ΔEkin=%.5f +- %.5f'%(ΔEkin,δΔEkin))
    calibrationFrame.loc[run,'ΔEkin']=ΔEkin; calibrationFrame.loc[run,'ΔEkin_uncertainty']=δΔEkin
    spectrumFrame = sh.loadSpectrumFrame(mass, targetDirectoryName)
    calibrationFrame.loc[run,'avgScanTime'] = np.mean(spectrumFrame['avgScanTime'])-scanTimeOffset

  calibrationScanTimes=np.array(calibrationFrame['avgScanTime'])
  calibrationVsScanNumber = bea.getCalibrationFunction(v0, δv0, calibrationFrame, np.array(calibrationFrame.index),mass, laserFreq, freqOffset)
  calibrationVsScanTime   = bea.getCalibrationFunction(v0, δv0, calibrationFrame, calibrationScanTimes, mass, laserFreq, freqOffset)
  
  '''exporting calibration results for analysis comparision purposes'''
  exportsPrefix='./'+directoryPrefix+'/CalibrationDiagnostics/'
  if energyCorrection==False:
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

    plt.title('calibration run centroids'); plt.xlabel('run'); plt.ylabel('centroid (MHz) - %.2f THz'%(freqOffset/1E6))
    plt.errorbar(calibrationFrame.index, y=calibrationFrame['centroid'], yerr=calibrationFrame['cent_uncertainty'], fmt='k.', label='individual runs')
    plt.gca().axhline(y=np.mean(calibrationFrame['centroid']), label='avg centroid')
    plt.legend(loc='best')
    plt.savefig(exportsPrefix+'/centroidVsRun_energyCorrected.png');plt.close()
  return(calibrationFrame, calibrationVsScanNumber, calibrationVsScanTime)

def fullAnalysis(a_ratio_fixed = True, equal_fwhm = False, cec_sim_toggle = "27Al_CEC_peaks.csv", spinList22=[4], peakModel='pseudoVoigt',whatToRun=False):
  if whatToRun==False: wtr=WhatToRun()#default to settings where everything runs.
  else: wtr=whatToRun

  beamEnergyAnalysisResults = bea.main(equal_fwhm = equal_fwhm, cec_sim_data_path = cec_sim_toggle, redoFits=wtr.fitAndLogToggle_BEA); #print(beamEnergyAnalysisResults)
  v0 = beamEnergyAnalysisResults[0]; δv0 = np.sqrt(beamEnergyAnalysisResults[1]**2+beamEnergyAnalysisResults[2]**2)
  # print(v0, δv0); quit()
  keV2amu=0.000001073544664258
  colinearity = False
  μ27Al=3.64070

  allIsotopesFrame = pd.DataFrame()
  '''starting with calibration runs'''
  directoryPrefix='results'+'/equal_fwhm_'+str(equal_fwhm)+'/cec_sim_toggle_'+str(cec_sim_toggle!=False)
  calibrationFrame_beforeBEC, calibrationVsScanNumber, calibrationVsScanTime = calibrationProcedure(runsDictionary[27],v0,δv0,exportSpectrumToggle=wtr.exportSpectrumToggle_calibration, directoryPrefix=directoryPrefix,
                                              fitAndLogToggle=wtr.fitAndLogToggle_calibration, equal_fwhm = equal_fwhm, cec_sim_toggle = cec_sim_toggle,freqOffset=freqOffset, scanTimeOffset=scanTimeOffset, peakModel=peakModel)

  calibrationFrame, _, _ = calibrationProcedure(runsDictionary[27],v0,δv0,exportSpectrumToggle=wtr.exportSpectrumToggle_calibration_bec, directoryPrefix=directoryPrefix,
                                              fitAndLogToggle=wtr.fitAndLogToggle_calibration_bec, equal_fwhm = equal_fwhm, cec_sim_toggle = cec_sim_toggle,freqOffset=freqOffset,energyCorrection=calibrationVsScanTime, scanTimeOffset=scanTimeOffset, peakModel=peakModel)

  #print(calibrationFrame)
  aRatio,uncertainty_Aratio1, uncertainty_Aratio2 = bea.weightedStats(calibrationFrame['aRatio'],calibrationFrame['aRatio_uncertainty'])
  uncertainty_Aratio=(uncertainty_Aratio1**2+uncertainty_Aratio2**2)**0.5
  fixed_Aratio=aRatio if a_ratio_fixed else False

  print('calibration frame aRatio, before energy corrections:',np.mean(calibrationFrame_beforeBEC['aRatio']))
  print('calibration frame aRatio, after energy corrections:', np.mean(calibrationFrame['aRatio']));

  def logEnergyCorrectionsVsRun(path, massNumber, runs, corrections):
    frame=pd.DataFrame({'runs':runs, 'correction':corrections})
    frame.to_csv(path+f"/calibrationVsRunNumberForMass{massNumber}.csv", index=False)
    
  '''now for all the isotopes'''
  spShiftEstimatesVolts=[]; spPropEstimates=[] #May 29, 2025: I will use the free_sp results from Al 25&23 to constrain the even isotopes.
  spShiftErrorsMHz=[] ; spPropErrors=[]
  exportsPrefix='./'+directoryPrefix+'/CalibrationDiagnostics/'
  for massNumber in [27,25,23,24,22]:#np.array(list(massDictionary.keys()))[::-1]:
    spScaleable=False
    scanDirec=str(massNumber)+'Al'
    laserFreq=3E6*laserDictionary[massNumber]
    tofWindow=tofDictionary[massNumber]#timeStepDictionary[massNumber]#
    runs = runsDictionary[massNumber]
    energyCorrectionToLoad_time = calibrationVsScanTime 
    energyCorrectionToLoad_runs = [float(calibrationVsScanNumber.eval(x=runNumber)) for runNumber in runs] #TODO: try this out and compare
    logEnergyCorrectionsVsRun(exportsPrefix, massNumber, runs, energyCorrectionToLoad_runs)
    energyCorrectionToLoad=energyCorrectionToLoad_time
    mass=massDictionary[massNumber]
    mass_uncertainty = massUncertDictionary[massNumber]
    targetDirectoryName = 'allScans_GroundMass/'
    print('mass%d'%massNumber)
    directoryPrefix='results'+'/equal_fwhm_'+str(equal_fwhm)+'/cec_sim_toggle_'+str(cec_sim_toggle!=False)+'/fixed_Aratio_'+str(a_ratio_fixed)
    if not os.path.exists(directoryPrefix): os.makedirs(directoryPrefix)
    args=[scanDirec, runs, laserFreq, mass, targetDirectoryName, iNucDictionary[massNumber], jGround, jExcited]
    kwargs={'fitAndLog':wtr.fitAndLogToggleDic[massNumber],'colinearity':colinearity, 'freqOffset':freqOffset, 'timeOffset':scanTimeOffset,  'tofWindow':tofWindow, 'cuttingColumn':'ToF',#'cuttingColumn':'time_step',#
    'cec_sim_data_path':cec_sim_toggle,'equal_fwhm':equal_fwhm, 'fixed_Aratio':fixed_Aratio,'directoryPrefix':directoryPrefix,'peakModel':peakModel,'spScaleable':spScaleable}
    
    if massNumber==24:
      bootStrappingDictionary={}
      bootStrappingDictionary['fixed_Aratio'] = [fixed_Aratio, uncertainty_Aratio]
      spShift=voltageShiftToFrequencyShift(mass, 3E6*laserDictionary[massNumber], nominalVoltage, 
                                                          np.mean(spShiftEstimatesVolts), colinearity=colinearity)
      _,_,spShiftErr=bea.weightedStats([voltageShiftToFrequencyShift(mass, 3E6*laserDictionary[massNumber], nominalVoltage, 
                                                          est, colinearity=colinearity) for est in spShiftEstimatesVolts], spShiftErrorsMHz)
      spProp=np.mean(spPropEstimates); 
      _,_,spPropErr=bea.weightedStats(spPropEstimates, spPropErrors)
      kwargs['fixed_spShift']=spShift; kwargs['fixed_spProp']=spProp
      # bootStrappingDictionary['fixed_spShift'] = [spShift, spShiftErr]
      # bootStrappingDictionary['fixed_spProp'] =  [spProp, spPropErr]
      # kwargs['bootStrappingDictionary']=bootStrappingDictionary
      '''first analyze data transformed wrt ground state nuclear mass'''
      processData(*args, exportSpectrum=wtr.exportSpectrumToggle, energyCorrection=False, **kwargs)
      processData(*args, exportSpectrum=wtr.exportSpectrumToggle_bec, energyCorrection=energyCorrectionToLoad, **kwargs)
      spectrumFrame = sh.loadSpectrumFrame(mass, targetDirectoryName); avgScanTime=np.mean(spectrumFrame['avgScanTime']) - scanTimeOffset
      uncorrectedResult = loadFitResults(directoryPrefix, massNumber, targetDirectoryName, energyCorrection=False)
      result = loadFitResults(directoryPrefix, massNumber, targetDirectoryName, energyCorrection=energyCorrectionToLoad)
      allIsotopesFrame = pd.concat([allIsotopesFrame, populateFrame(mass, mass_uncertainty, iNucDictionary[massNumber][0], result, prefix='iso0', uncorrectedResult=uncorrectedResult, scanTime=avgScanTime)], ignore_index=True)
      '''and now for isomer'''
      mass=massDictionary[massNumber]+keV2amu*425.81; mass_uncertainty = keV2amu*0.1 #isomer excited by 425.81 (10) keV
      targetDirectoryName = 'allScans_IsomerMass/'
      args=[scanDirec, runs, laserFreq, mass, targetDirectoryName, iNucDictionary[massNumber], jGround, jExcited]
      processData(*args, exportSpectrum=wtr.exportSpectrumToggle, energyCorrection=False, **kwargs)
      processData(*args, exportSpectrum=wtr.exportSpectrumToggle_bec, energyCorrection=energyCorrectionToLoad, **kwargs)
      result = loadFitResults(directoryPrefix, massNumber, targetDirectoryName, energyCorrection=energyCorrectionToLoad)
      spectrumFrame = sh.loadSpectrumFrame(mass, targetDirectoryName); avgScanTime=np.mean(spectrumFrame['avgScanTime']) - scanTimeOffset
      uncorrectedResult = loadFitResults(directoryPrefix, massNumber, targetDirectoryName, energyCorrection=False)
      allIsotopesFrame = pd.concat([allIsotopesFrame, populateFrame(mass, mass_uncertainty, iNucDictionary[massNumber][1], result, prefix='iso1', uncorrectedResult=uncorrectedResult, scanTime=avgScanTime)], ignore_index=True)     

      #TODO:callBootstrapper(*args,energyCorrection=energyCorrectionToLoad, **kwargs, bootStrappingDictionary=bootStrappingDictionary)
    elif massNumber==22:
      kwargs['spShiftGuess']=voltageShiftToFrequencyShift(mass, 3E6*laserDictionary[massNumber], nominalVoltage, 
                                                          np.mean(spShiftEstimatesVolts), colinearity=colinearity);
      kwargs['fixed_spShift']=True
      kwargs['spPropGuess']=np.mean(spPropEstimates); kwargs['fixed_spProp']=True
      for i in spinList22:
        print('spin22=%d'%i)
        args=[scanDirec, runs, laserFreq, mass, targetDirectoryName, [i], jGround, jExcited]
        processData(*args, **kwargs, exportSpectrum=wtr.exportSpectrumToggle, energyCorrection=False, subPath='I=%d'%i, keepLessIntegratedBins=False)
        processData(*args, **kwargs, exportSpectrum=wtr.exportSpectrumToggle_bec, energyCorrection=energyCorrectionToLoad, subPath='I=%d'%i, keepLessIntegratedBins=False)
        spectrumFrame = sh.loadSpectrumFrame(mass, targetDirectoryName); avgScanTime=np.mean(spectrumFrame['avgScanTime']) - scanTimeOffset
        uncorrectedResult = loadFitResults(directoryPrefix, massNumber, targetDirectoryName, energyCorrection=False, subPath='I=%d'%i)
        result = loadFitResults(directoryPrefix, massNumber, targetDirectoryName, energyCorrection=energyCorrectionToLoad, subPath='I=%d'%i)
        allIsotopesFrame = pd.concat([allIsotopesFrame, populateFrame(mass, mass_uncertainty, i, result, prefix='iso0', uncorrectedResult=uncorrectedResult, scanTime=avgScanTime)], ignore_index=True)

    elif massNumber==27:
      stable_directoryPrefix='results'+'/equal_fwhm_'+str(equal_fwhm)+'/cec_sim_toggle_'+str(cec_sim_toggle!=False)
      stableFrame=pd.DataFrame()
      for run in runsDictionary[27]:
        stable_targetDirectoryName = 'Scan%d'%run
        spectrumFrame = sh.loadSpectrumFrame(mass, targetDirectoryName); avgScanTime=np.mean(spectrumFrame['avgScanTime']) - scanTimeOffset
        uncorrectedResult = loadFitResults(stable_directoryPrefix, massNumber, stable_targetDirectoryName, energyCorrection=False)
        result = loadFitResults(stable_directoryPrefix, massNumber, stable_targetDirectoryName, energyCorrection=energyCorrectionToLoad)
        stableFrame = pd.concat([stableFrame, populateFrame(mass, mass_uncertainty, iNucDictionary[massNumber][0], result, prefix='iso0', uncorrectedResult=uncorrectedResult, scanTime=avgScanTime)], ignore_index=True)
      stable_aLower, unc1, unc2= bea.weightedStats(stableFrame['aLower'], stableFrame['aLower_uncertainty']); stable_aLower_uncertainty = np.sqrt(unc1**2+unc2**2)
      stable_aUpper, unc1, unc2= bea.weightedStats(stableFrame['aUpper'], stableFrame['aUpper_uncertainty']); stable_aUpper_uncertainty = np.sqrt(unc1**2+unc2**2)
      stable_aRatio=stable_aLower/stable_aUpper
      stable_aRatio_uncertainty = (stable_aRatio**2) *( (stable_aLower_uncertainty/stable_aLower)**2 + (stable_aUpper_uncertainty/stable_aUpper)**2)
      #stable_centroid, unc1, unc2= bea.weightedStats(stableFrame['centroid'], stableFrame['cent_uncertainty']) ; stable_centroid_uncertainty = np.sqrt(unc1**2+unc2**2)
      stable_centroid, stable_centroid_uncertainty = v0-freqOffset,δv0
      stable_uncorrectedCentroid, unc1, unc2= bea.weightedStats(stableFrame['uncorrectedCentroid'], stableFrame['uncorrectedCentroid_uncertainty']); stable_uncorrectedCentroid_uncertainty = np.sqrt(unc1**2+unc2**2)
      stableDict={
      'massNumber':round(mass),'mass':[mass], 'mass_uncertainty':[mass_uncertainty],"I":[iNucDictionary[massNumber][0]],
      "aLower":[stable_aLower],"aLower_uncertainty":[stable_aLower_uncertainty],
      "aUpper":[stable_aUpper],"aUpper_uncertainty":[stable_aUpper_uncertainty],
      "aRatio":[stable_aRatio],"aRatio_uncertainty":[stable_aRatio_uncertainty],
      "centroid":[stable_centroid],"cent_uncertainty":[stable_centroid_uncertainty],
      "uncorrectedCentroid":[stable_uncorrectedCentroid],"uncorrectedCentroid_uncertainty":[stable_uncorrectedCentroid_uncertainty],
      'avgScanTime':avgScanTime}
      allIsotopesFrame = pd.concat([allIsotopesFrame, pd.DataFrame(stableDict)], ignore_index=True)

    else:
      processData(*args, **kwargs, exportSpectrum=wtr.exportSpectrumToggle, energyCorrection=False)
      processData(*args, **kwargs, exportSpectrum=wtr.exportSpectrumToggle_bec, energyCorrection=energyCorrectionToLoad)
      spectrumFrame = sh.loadSpectrumFrame(mass, targetDirectoryName); avgScanTime=np.mean(spectrumFrame['avgScanTime']) - scanTimeOffset
      uncorrectedResult = loadFitResults(directoryPrefix, massNumber, targetDirectoryName, energyCorrection=False)
      result = loadFitResults(directoryPrefix, massNumber, targetDirectoryName, energyCorrection=energyCorrectionToLoad)  
      allIsotopesFrame = pd.concat([allIsotopesFrame, populateFrame(mass, mass_uncertainty, iNucDictionary[massNumber][0], result, prefix='iso0', uncorrectedResult=uncorrectedResult, scanTime=avgScanTime)], ignore_index=True)
      spPropEstimates+=[result['iso0_spProp'].value]; spPropErrors+=[result['iso0_spProp'].stderr]
      spShiftEstimatesVolts+=[freqShiftToVoltageShift(mass, 3E6*laserDictionary[massNumber], result['iso0_centroid'].value,
                                                      result['iso0_spShift'].value, freqOffset=freqOffset)]
      spShiftErrorsMHz+=[result['iso0_spShift'].stderr]
      #print(allIsotopesFrame[['massNumber', 'centroid']]); quit()

  def beamEnergyBootstrapping(v0, δv0, calibrationFrame, xEval, mass, laserFreq, freqOffset, numTrials):
    yEvals = np.zeros((len(xEval),numTrials))
    for i in range(numTrials):
      v_sample = np.random.normal(loc=v0, scale=δv0)
      res=bea.getCalibrationFunction(v_sample, δv0, calibrationFrame, xEval, mass, laserFreq, freqOffset, randomSampling=True)
      yEvals[:,i] = res.eval(x=xEval)
    return(yEvals)

  def isotopeShiftBootstrapping(v0, δv0, calibrationFrame, scanTimeOffset, isotopesFrame, numTrials, massDictionary, laserDictionary, freqOffset, referenceMassNumber):
    isotopesList = list(isotopesFrame.index)
    energyCorrectedCentroidsSamples = np.zeros((len(isotopesList),numTrials))
    calibrationScanTimes=np.array(calibrationFrame['avgScanTime'])
    calibrationMass=massDictionary[referenceMassNumber]
    calibrationLaserFreq=3E6*laserDictionary[referenceMassNumber]
    for i in range(numTrials):
      v_sample = np.random.normal(loc=v0, scale=δv0)
      res=bea.getCalibrationFunction(v_sample, δv0, calibrationFrame, calibrationScanTimes, calibrationMass, calibrationLaserFreq, freqOffset, randomSampling=True)
      beamEnergies = res.eval(x=isotopesFrame['avgScanTime'])
      for j in isotopesList:
        mass=isotopesFrame.loc[j,'mass']; massNumber=isotopesFrame.loc[j,'massNumber']
        centroid = isotopesFrame.loc[j,'uncorrectedCentroid']+freqOffset; laserFreq=3E6*laserDictionary[massNumber]
        energyCorrectedCentroidsSamples[j,i] = propogateBeamEnergyCorrectionToCentroid(mass, centroid, laserFreq, beamEnergies[j]) - freqOffset
    isotopeShiftsSamples = energyCorrectedCentroidsSamples - energyCorrectedCentroidsSamples[0,:]
    results = {}
    for j in isotopesList:
      mass=isotopesFrame.loc[j,'mass']; massNumber=isotopesFrame.loc[j,'massNumber']
      results[j]=[massNumber, mass, isotopeShiftsSamples[j,:]]
    return(results)

  def chargeRadiusBootstrapping(v0, δv0, calibrationFrame, scanTimeOffset, isotopesFrame, numTrials, massDictionary, laserDictionary, freqOffset, referenceMassNumber, K, F):
    chargeRadiusDic={}
    isotopeShiftSamplesDic = isotopeShiftBootstrapping(v0, δv0, calibrationFrame, scanTimeOffset, isotopesFrame, numTrials, massDictionary, laserDictionary, freqOffset, referenceMassNumber)

    assert isotopeShiftSamplesDic[0][0]==referenceMassNumber #quick sanity check so I don't have to go searching through this
    referenceMass = isotopeShiftSamplesDic[0][1]
    chargeRadiusDic[0] = [referenceMassNumber, referenceMass, np.zeros(numTrials)]
    for i in list(isotopeShiftSamplesDic.keys())[1:]:
      compMassNumber=isotopeShiftSamplesDic[i][0]
      compMass = isotopeShiftSamplesDic[i][1]; massFactor=1/compMass-1/referenceMass
      isoShifts = isotopeShiftSamplesDic[i][2]
      δrsq=(isoShifts-K*massFactor)/F
      chargeRadiusDic[i] = [compMassNumber,compMass, δrsq]

    # for index in δν.index:
    #   massFactor=1/δν.loc[index]['mass']-1/δν.loc[stableIndex]['mass']
    #   δν.loc[index,'δrsq']=(δν.loc[index]['shift']-K*massFactor)/F
    return(chargeRadiusDic)

  isoTimes=np.array(allIsotopesFrame['avgScanTime'])
  calibrationScanTimes=np.array(calibrationFrame['avgScanTime'])
  
  #section for estimating error in beam energy corrections  
  t0=time.time()
  yEvals = beamEnergyBootstrapping(v0, δv0, calibrationFrame, calibrationScanTimes, massDictionary[27], 3E6*laserDictionary[27], freqOffset, 200)
  t1=time.time()
  #print('time elapsed',t1-t0)

  # for i in range(len(isoTimes)):
  #   plt.plot(isotopeShiftsSamples[i][2], label=allIsotopesFrame.loc[i,'massNumber']);
  # plt.xlabel('trial i'); plt.ylabel('isotope shift from calibration fit for bootstrap trial i'); plt.title('Isotopes Shifts From Random Sampling Beam Energy Corrections')
  # plt.legend()
  # plt.close()

  # for i in range(len(isoTimes)):
  #   plt.plot(yEvals[i,:], label=allIsotopesFrame.loc[i,'massNumber']);
  # plt.xlabel('trial i'); plt.ylabel('beam energy corrections from calibration fit for bootstrap trial i'); plt.title('Random Sampling Beam Energy Corrections')
  # plt.legend()
  # plt.show()

  # convergenceArray=[]
  # nSamples = np.array(list(range(2,20)) + list(range(20,250,5)));
  # for n in nSamples:
  #   print('n=%d'%n)
  #   yEvals = beamEnergyBootstrapping(v0, δv0, calibrationFrame, calibrationScanTimes, massDictionary[27], 3E6*laserDictionary[27], freqOffset, n)
  #   convergenceArray+=[np.std(yEvals)]

  # plt.plot(nSamples, convergenceArray)
  # plt.xlabel('num Samples'); plt.ylabel(r'Beam energy correction $(\langle t\rangle_{\text{all scans}})$'); plt.title('Convergence of error estimate for beam energy corrections')
  # plt.show()

  # convergenceArray=[[] for i in allIsotopesFrame.index]
  # nSamples = np.array(list(range(2,20)) + list(range(20,250,5)));
  # for n in nSamples:
  #   print('n=%d'%n)
  #   isotopeShiftsSamples = isotopeShiftBootstrapping(v0, δv0, calibrationFrame, scanTimeOffset, allIsotopesFrame, n, massDictionary, laserDictionary, freqOffset, 27)
  #   for i in allIsotopesFrame.index:
  #     convergenceArray[i]+=[np.std(isotopeShiftsSamples[i][2])]

  # for i in allIsotopesFrame.index:
  #   plt.plot(nSamples, convergenceArray[i], label=isotopeShiftsSamples[i][0])
  # plt.xlabel('num Samples'); plt.ylabel(r'Beam energy corrected  $δv_{i,27}$(MHz)'); plt.title('Convergence of error estimate for isotope shifts from beam energy correction')
  # plt.legend(loc=1)
  # plt.show()

  
  isotopeShiftsDic = isotopeShiftBootstrapping(v0, δv0, calibrationFrame, scanTimeOffset, allIsotopesFrame, 200, massDictionary, laserDictionary, freqOffset, 27)

  print('charge radius scatter from bootstrapping beam energy correction procedure')
  for i in list(isotopeShiftsDic.keys()):
    plt.plot(isotopeShiftsDic[i][2], label=isotopeShiftsDic[i][0]);
    print('mass%d'%isotopeShiftsDic[i][0], 'δv=%.5f'%np.mean(isotopeShiftsDic[i][2]), '+/-', np.std(isotopeShiftsDic[i][2]));
    allIsotopesFrame.loc[i,'shift_uncertainty_BEC'] = np.std(isotopeShiftsDic[i][2])
  plt.xlabel('trial i'); plt.ylabel('isotope shift from calibration fit for bootstrap trial i'); plt.title('Isotope Shifts From Random Sampling Beam Energy Corrections')
  plt.legend()
  plt.close();

  stableIndex = allIsotopesFrame.loc[allIsotopesFrame['massNumber']==27].index[0]

  kα_Skrip=-0.7*1000; σK_Skrip=2.1*1000 #Total MassShift sensitivity in MHz*amu
  Fα_Skrip=70.11;     σF_Skrip=0.13 #FieldShift sensitivity in MHz/fm^2

  chargeRadiusDic = chargeRadiusBootstrapping(v0, δv0, calibrationFrame, scanTimeOffset, allIsotopesFrame, 200, massDictionary, laserDictionary, freqOffset, 27, kα_Skrip, Fα_Skrip)

  print('charge radius scatter from bootstrapping beam energy correction procedure')
  for i in list(chargeRadiusDic.keys()):
    plt.plot(chargeRadiusDic[i][2], label=chargeRadiusDic[i][0]);
    print('mass%d'%chargeRadiusDic[i][0], 'δrsq=%.5f'%np.mean(chargeRadiusDic[i][2]), '+/-', np.std(chargeRadiusDic[i][2]));
  plt.xlabel('trial i'); plt.ylabel('charge radii from calibration fit for bootstrap trial i'); plt.title('Charge Radii From Random Sampling Beam Energy Corrections')
  plt.legend()
  plt.close()

  def extractChargeRadii(K, F,σK,σF, δν, stableIndex):
    for index in δν.index:
      mi=δν.loc[index]['mass']; mRef=δν.loc[stableIndex]['mass']
      δmi=δν.loc[index]['mass_uncertainty']; δmRef=δν.loc[stableIndex]['mass_uncertainty']
      massFactor=1/mi-1/mRef
      δν.loc[index,'δrsq']=(δν.loc[index]['shift']-K*massFactor)/F
      uncert1=(1/F**2) * δν.loc[index]['shift_uncertainty_fit']**2 
      uncert2=(1/F**2) * δν.loc[index]['shift_uncertainty_BEC']**2
      uncert3=((massFactor/F)**2) * σK**2
      uncert4=(((δν.loc[index]['shift']-K*massFactor)/F**2)**2) * σF**2
      uncert5=((K/F)**2)*((δmi/(mi**2))**2 + (δmRef/(mRef**2))**2)
      if index==stableIndex:
        δν.loc[index,'δrsq_uncertainty_fit']=0
        δν.loc[index,'δrsq_uncertainty_BEC']=0
        δν.loc[index,'δrsq_uncertainty_mass_factor']=0
        δν.loc[index,'δrsq_uncertainty_field_factor']=0
        δν.loc[index,'δrsq_uncertainty_mass_measure']=0
        δν.loc[index,'δrsq_uncertainty_exp']=0
        δν.loc[index,'δrsq_uncertainty_theory']=0
        δν.loc[index,'δrsq_uncertainty_total']=0
      else:
        δν.loc[index,'δrsq_uncertainty_fit']=np.sqrt(uncert1)
        δν.loc[index,'δrsq_uncertainty_BEC']=np.sqrt(uncert2)
        δν.loc[index,'δrsq_uncertainty_mass_factor']=np.sqrt(uncert3)
        δν.loc[index,'δrsq_uncertainty_field_factor']=np.sqrt(uncert4)
        δν.loc[index,'δrsq_uncertainty_mass_measure']=np.sqrt(uncert5)
        δν.loc[index,'δrsq_uncertainty_exp']=np.sqrt(uncert1+uncert2)
        δν.loc[index,'δrsq_uncertainty_theory']=np.sqrt(uncert3+uncert4+uncert5)
        δν.loc[index,'δrsq_uncertainty_total']=np.sqrt(uncert1+uncert2+uncert3+uncert4+uncert5)
    return()

  xData =  np.array(calibrationFrame['avgScanTime']).astype(float) ; yData = np.array(calibrationFrame['centroid']).astype(float)
  allIsotopesFrame['interpolatedReference']=np.interp(allIsotopesFrame['avgScanTime'],xData, yData)
  allIsotopesFrame['shift']=allIsotopesFrame['centroid']-allIsotopesFrame.loc[stableIndex]['centroid']
  # allIsotopesFrame['shift']=allIsotopesFrame['centroid']-allIsotopesFrame['interpolatedReference']
  allIsotopesFrame['shift_uncertainty_fit'] = np.sqrt(allIsotopesFrame['cent_uncertainty']**2+allIsotopesFrame.loc[stableIndex]['cent_uncertainty']**2)
  allIsotopesFrame.loc[stableIndex,'shift_uncertainty_fit']=0

  extractChargeRadii(kα_Skrip,Fα_Skrip,σK_Skrip,σF_Skrip, allIsotopesFrame, stableIndex)
  

  μCommonFactor = μ27Al/allIsotopesFrame.loc[stableIndex]['I']/allIsotopesFrame.loc[stableIndex]['aLower']
  allIsotopesFrame['μ']=allIsotopesFrame['aLower']*allIsotopesFrame['I']*μCommonFactor
  allIsotopesFrame['μ_uncertainty']=allIsotopesFrame['aLower_uncertainty']*allIsotopesFrame['I']*μCommonFactor

  # pd.set_option('display.max_columns', 8)
  # print(allIsotopesFrame[['massNumber','I', "uncorrectedCentroid", "uncorrectedCentroid_uncertainty"]])
  # print(allIsotopesFrame[['massNumber','I', 'shift','shift_uncertainty_fit','shift_uncertainty_BEC']])
  # print(allIsotopesFrame[['massNumber','I', 'δrsq','δrsq_uncertainty_fit','δrsq_uncertainty_BEC','δrsq_uncertainty_mass_factor','δrsq_uncertainty_field_factor','δrsq_uncertainty_total']])
  # print(allIsotopesFrame[['massNumber','I', 'aLower','aLower_uncertainty','aUpper','aUpper_uncertainty','μ']])

  nameTag='equal_fwhm_'+str(equal_fwhm)+'-cec_sim_toggle_'+str(cec_sim_toggle!=False)+'-fixed_Aratio_'+str(a_ratio_fixed)+'_'
  allIsotopesFrame.to_csv(directoryPrefix+'/'+nameTag+'CompiledAnalysisResults.csv')#, header='#cec_sim: '+str(cec_sim_toggle)+'; equal_fwhm: '+str(equal_fwhm)+'; fixed_Aratio:'+str(fixed_Aratio))
  return(allIsotopesFrame)

  # for i in allIsotopesFrame.index:
  #   m = allIsotopesFrame.loc[i, 'mass'];
  #   mu, sigma_k = bea.bootstrapUncertainty(extractChargeRadius, 1000, [[kα_Skrip, σK_Skrip], [Fα_Skrip, 0], [m,0], [allIsotopesFrame.loc[i, 'shift'],0] ])
  #   mu, sigma_F = bea.bootstrapUncertainty(extractChargeRadius, 1000, [[kα_Skrip, 0], [Fα_Skrip, σF_Skrip], [m,0], [allIsotopesFrame.loc[i, 'shift'],0] ])
  #   print('mass = %.3f'%m, 'sigma_k = %.3f'%sigma_k, 'sigma_F = %.5f'%sigma_F)

def extractChargeRadius(K, F, m, δν):
  mass27=26.98153841; massFactor=1/m-1/mass27
  δrsq=(δν-K*massFactor)/F; return(δrsq)

def analysisPlot(allIsotopesFrame,label='RISE Data',color='green',posibleSpins=False):
  theoryData=np.loadtxt("theoryData_Al.csv", skiprows=1,usecols=9,delimiter=','); xData=[22,23,24,25,26,27]
  emDat = theoryData[0::3]; emResult = (emDat**2-emDat[-1]**2); plt.plot(xData, emResult, '^', linestyle='dashed',markeredgecolor= "black", markersize=8,color='yellow', label='1.8/2.0 (EM)'); 
  n2Dat = theoryData[1::3]; n2Result = (n2Dat**2-n2Dat[-1]**2); plt.plot(xData, n2Result, 'D', linestyle='dashed',markeredgecolor= "black", markersize=8,color='orange', label=r'N$^2$LO$_{\text{GO}}$')
  n3Dat = theoryData[2::3]; n3Result = (n3Dat**2-n3Dat[-1]**2); plt.plot(xData, n3Result, 's', linestyle='dashed',markeredgecolor= "black", markersize=8,color='purple',label=r'N$^3$LO EM500')

  groundNucleiFrame=allIsotopesFrame.query('not(massNumber==24 and I==1) and not(massNumber==22 and not I==4)')
  plt.plot(groundNucleiFrame['mass'], groundNucleiFrame['δrsq'], color=color)
  plt.errorbar(groundNucleiFrame['mass'], y=groundNucleiFrame['δrsq'],yerr=groundNucleiFrame['δrsq_uncertainty_fit'], fmt='.',color=color, markersize=10, label=label)
  tempFrame=allIsotopesFrame.query('massNumber==24 and I==1')
  plt.plot(tempFrame['mass'], tempFrame['δrsq'], '*',color=color, label='isomeric state')

  if posibleSpins:
    iColors=['brown','red','orange','green','blue','purple']
    for i in range(2,7):
      tempFrame=allIsotopesFrame.query('massNumber==22 and I==%d'%i)
      plt.errorbar(tempFrame['mass'], y=tempFrame['δrsq'],yerr=tempFrame['δrsq_uncertainty_fit'], fmt='.', color=iColors[i-1])#, label='I=%d'%i)
  plt.ylim([-.25,.35])
  #handles, labels = plt.gca().get_legend_handles_labels()
  #order = [0,1,2,4,3]
  #plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc=1)
  plt.xlabel("A")
  plt.ylabel(r'$\delta\langle r^2\rangle_{ch}^{A,27}$')
  plt.title("Aluminum Charge Radius Predictions - Temporal Calibration Corrected")
  #plt.close()

if __name__ == '__main__':pass
  # peakModel='pseudoVoigt'
  # equal_fwhm_toggle_list = [True]#,False]#False,
  # cec_sim_toggle_list = [False, "27Al_CEC_peaks.csv"]#,
  # a_ratio_fixed_list = [True,False]
  # i=0
  # allFramesDic={}
  # for equal_fwhm_toggle in equal_fwhm_toggle_list:
  #   for cec_sim_toggle in cec_sim_toggle_list:
  #     print(equal_fwhm_toggle,cec_sim_toggle)
  #     for a_ratio_toggle in a_ratio_fixed_list:
  #       allIsotopesFrame=fullAnalysis(a_ratio_fixed = a_ratio_toggle, equal_fwhm = equal_fwhm_toggle, cec_sim_toggle = cec_sim_toggle, spinList22=[4], peakModel=peakModel)
  #       allFramesDic[('fwhm_'+str(equal_fwhm_toggle),'cec_'+str(cec_sim_toggle), 'aRatio_'+str(a_ratio_toggle))]=allIsotopesFrame
  #       i+=1
  #       print('i=',i)
  # print(allIsotopesFrame)