import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, pickle
import spectrumHandler as sh
import hyperfinePredictorGREAT as hpg

class Spectrum:
  '''This is a class to store everything associated with a spectrum'''
  essentialKeys=['scanDirectory', 'directoryPrefix', 'runs', 'laserFrequency', 'mass', 'targetDirectory', 'nuclearSpinList', 'colinearity', 'jGround', 'jExcited']

  def __init__(self, constructSpectrum=False, energyCorrection=False, mass_uncertainty=0, timeOffset=0, **kwargs):
    self.__dict__.update(kwargs);
    for key in Spectrum.essentialKeys: 
      if not (key in self.__dict__):
        print(f'Error: no {key} provided. All of the following kwargs must be passed to construct a Spectrum object\n{Spectrum.essentialKeys}')
        raise Exception
    if energyCorrection: #If an energy correction is provided, the created object will also keep track of the uncorrected results
      self.uncorrectedSpectrum = Spectrum(constructSpectrum=False, energyCorrection=False, mass_uncertainty=mass_uncertainty, **kwargs)
    self.energyCorrection=energyCorrection
    self.mass_uncertainty=mass_uncertainty
    self.timeOffset=timeOffset
    self.resultsPath=f'./{self.directoryPrefix}/mass{round(self.mass)}/{self.targetDirectory}/' + ('energyCorrected/' if self.energyCorrection else '')
    self.suffix='_energyCorrected' if self.energyCorrection else ''
    self.setSpectrum(constructSpectrum=constructSpectrum)# self.spectralData = {}

  def setSpectrum(self, constructSpectrum=False):
    spectrumPath = self.resultsPath+'spectralData_energyCorrected.csv' if self.energyCorrection else self.resultsPath+'spectralData.csv'
    if not os.path.exists(spectrumPath): print(f"can't find {spectrumPath}")
    if not os.path.exists(spectrumPath) or constructSpectrum: sh.exportSpectrumFrame(laserFreq=self.laserFrequency,targetDirectoryName=self.targetDirectory, scanDirec=self.scanDirectory, **self.__dict__)
    spectrumFrame = sh.loadSpectrumFrame(self.mass, self.targetDirectory, directoryPrefix=self.directoryPrefix, energyCorrection=self.energyCorrection)
    self.spectrumFrame=spectrumFrame
    self.x=np.array(spectrumFrame['dcf']).copy(); self.y=np.array(spectrumFrame['countrate']).copy(); self.yUncertainty=np.array(spectrumFrame['uncertainty']).copy()
    #when all the frequencies are huge, lmfit struggles to return uncertainties. Subtracting off a recorded offset
    if 'frequencyOffset' not in self.__dict__: self.frequencyOffset=np.mean(self.x)
    self.x-=self.frequencyOffset

  def getSpectrum(self):
    if 'spectrumFrame' in self.__dict__: return self.spectrumFrame
    self.setSpectrum; return self.spectrumFrame

  def fitDat(self, **kwargs):
    '''fitParams: peakModel=peakModel, transitionLabel=transitionLabel, colinearity=colinearity, laserFreq=laserFreq,
                freqOffset=freqOffset, centroidGuess=centroidGuess, cec_sim_data_path=cec_sim_data_path, fixed_spShift=fixed_spShift,
                fixed_Alower=fixed_Alower, fixed_Aupper=fixed_Aupper, fixed_Aratio=fixed_Aratio, equal_fwhm=equal_fwhm,  weightsList=weightsList, fixed_Sigma=fixed_Sigma, fixed_Gamma=fixed_Gamma,**kwargs'''
    print(kwargs)
    kwargs['colinearity']=self.colinearity; kwargs['frequencyOffset']=self.frequencyOffset; kwargs['laserFrequency']=self.laserFrequency
    # result,interpolator = hpg.fitData(self.x, self.y, self.yUncertainty, self.mass, self.nuclearSpinList, self.jGround, self.jExcited, **kwargs)
    result,interpolator = self.fittingFunction(self.x, self.y, self.yUncertainty, **kwargs)
    result.fittingkwargs=kwargs
    return(result, interpolator)
  
  def plotFitResults(self, result, interpolator, **fittingkwargs):
    x_interp=np.linspace(np.min(self.x), np.max(self.x), 1000); y_interp = 0*x_interp
    for comp in interpolator:
      y_interp+=comp(x_interp)
    fig, (ax1, ax2) = plt.subplots(2, figsize=(16,9), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    plotTitle=str(round(self.mass))+'Al '+self.targetDirectory
    plotTitle +=' - colinear' if self.colinearity else ' - anticolinear'
    for key,val in fittingkwargs.items(): plotTitle+=f', {key}: {val}'
    if self.energyCorrection: plotTitle += ',beam energy corrected by '+str(self.energyCorrection)
    ax1.set_title(plotTitle)
    ax1.errorbar(self.x,self.y,yerr=self.yUncertainty,fmt='b.',ecolor='black',capsize=1,markersize=8, label='data')
    ax1.plot(x_interp, y_interp, 'b-', label='best') #ax1.plot(xData, result.init_fit, 'r--',alpha=0.25, label='init')
    if len(self.nuclearSpinList)>1:
      # components = result.eval_components(x=x_interp)  
      for k,iNuc in enumerate(self.nuclearSpinList):
        comp=interpolator[k+1]
        ax1.plot(x_interp, comp(x_interp), linestyle='--', label=f'I = {iNuc}')
    ax1.set_ylabel('countrate'); ax2.set_ylabel('residuals'); ax2.set_xlabel('(Frequency - ' +str(self.frequencyOffset)+')(MHz)')
    ax1.legend(loc=1)
    ax2.errorbar(self.x,(self.y-result.best_fit)/self.yUncertainty,yerr=1,fmt='b.',ecolor='black',capsize=1,markersize=8)
    fig.set_size_inches(18.5, 10.5)
    fig.savefig(f'{self.resultsPath}fit_plot{self.suffix}.png',dpi=100)
    fig.clf()
    plt.close(fig)

  def writePeakPositions(self, filename, result, **fittingkwargs):
    if 'cec_sim_data_path' in fittingkwargs.keys(): cec_sim_data_path=fittingkwargs['cec_sim_data_path']
    else: cec_sim_data_path = False
    with open(filename,'w') as file:
      file.write(f'scan frequency offset: {self.frequencyOffset}\n')
      file.write('Peak positions (MHz):\n')      
      for k,iNuc in enumerate(self.nuclearSpinList):
        centroid=result.params['iso0_centroid'].value
        A1,A2,B1,B2=result.params['iso'+str(k)+'_'+'Alower'].value, result.params['iso'+str(k)+'_'+'Aupper'].value,\
                    result.params['iso'+str(k)+'_'+'Blower'].value, result.params['iso'+str(k)+'_'+'Bupper'].value
        peakFreqs, _ = hpg.hfsLinesAndStrengths(iNuc,self.jGround,self.jExcited,A1,A2,B1=B1,B2=B2)+(centroid-self.frequencyOffset)
        peakFreqs= np.array(peakFreqs)+self.frequencyOffset
        file.write('nuclear state %d:\n\tpeakFreqs-offset:'%(k+1)+'[')
        for peakFreq in peakFreqs[:-1]: file.write(str(float(peakFreq-self.frequencyOffset))+', ')
        file.write(str(float(peakFreqs[-1]-self.frequencyOffset))+']\n')
        file.write('\tcentroid: '+str(centroid)); file.write('\n')
        for i,peakFreq in enumerate(peakFreqs):
          file.write('\tpeak %d: '%i+str(peakFreq)); file.write('\n')
          if cec_sim_data_path:
            cec_sim_data=np.loadtxt(cec_sim_data_path, skiprows=1,delimiter=',')
            cec_sim_energies = cec_sim_data[:,2]; sp_fractions=cec_sim_data[:,1]
            cec_sim_energies=cec_sim_energies[sp_fractions>0]; sp_fractions=sp_fractions[sp_fractions>0]; originalFractionList=sp_fractions
            sp_scaling_list=result.params['spScaling'].value*np.ones_like(sp_fractions); sp_scaling_list[0]=1
            sp_fractions=sp_fractions*sp_scaling_list
            sp_shifts, sp_fractions, broadeningList = hpg.generateSidePeaks(self.mass, self.laserFreq, peakFreq, originalFractionList, 
                                                      cec_sim_energies, freqOffset=0, colinearity=colinearity, cecBinning=5)
            file.write('\t\tsidepeak frequencies: '+str(sp_shifts)); file.write('\n')
            file.write('\t\tsidepeak fractions: '+str(sp_fractions)); file.write('\n')

          else:
            file.write('\t\tsidepeak fraction, spacing: '+str(result.params['iso0_spProp'].value)+', '+str(result.params['iso0_spShift'].value)); file.write('\n')

  def fitAndLogData(self, **fittingkwargs):
    fittingkwargs['colinearity']=self.colinearity
    fittingkwargs['frequencyOffset']=self.frequencyOffset
    fittingkwargs['laserFrequency']=self.laserFrequency
    if self.energyCorrection:
      if self.uncorrectedSpectrum.loadFitResults(): #this conditional checks for existence of uncorrected fit results, while simultaneously loading them to obj if they do exist
        if self.uncorrectedSpectrum.fittingkwargs!=fittingkwargs: 
          self.uncorrectedSpectrum.fitAndLogData(**fittingkwargs)
      else: self.uncorrectedSpectrum.fitAndLogData(**fittingkwargs)
    result,interpolator=self.fitDat(**fittingkwargs)

    with open(    f'{self.resultsPath}fit_report{self.suffix}.txt','w' ) as file:
      file.write(f'Frequency Offset used for fit: {self.frequencyOffset}\n')
      file.write(result.fit_report())
    for i in range(len(self.nuclearSpinList)): result.params[f"iso{i}_centroid"].value+=self.frequencyOffset
    statsDic = hpg.makeDictionaryFromFitStatistics(result)
    self.resultParams=result.params
    self.fittingkwargs=fittingkwargs
    self.fitStats=statsDic

    def fitEvaluator(xInt):
      if not os.path.exists(self.resultsPath+'fitEvaluator/'): os.mkdir(self.resultsPath+'fitEvaluator/')
      fittingkwargs=self.fittingkwargs.copy()
      if fittingkwargs['cec_sim_data_path']!=False:
        fittingkwargs['cec_sim_data']=np.loadtxt(cec_sim_data_path, skiprows=1,delimiter=',')
      else: fittingkwargs['cec_sim_data']=[]
      fittingkwargs['frequencyOffset']=self.frequencyOffset
      fittingkwargs['laserFrequency']=self.laserFrequency
      fittingkwargs['colinearity']=self.colinearity
      bgParmVals={'bg':result.params['bg'].value, 'slope':result.params['slope'].value}
      with open(f'{self.resultsPath}fitEvaluator/bgParms.pkl','wb') as file: pickle.dump(bgParmVals, file)
      functions=[hpg.backgroundFunction(x=xInt, **bgParmVals)]
      evalKwargsList=['equal_fwhm', 'cec_sim_data', 'frequencyOffset', 'laserFrequency', 'colinearity']
      evalKwargs = {kwarg:fittingkwargs[kwarg] for kwarg in evalKwargsList}
      with open(f'{self.resultsPath}fitEvaluator/evalKwargs.pkl','wb') as file: pickle.dump(evalKwargs, file)
      for comp in result.components[1:]:
        evalParms={}
        pref=comp.prefix
        for param_name, value in result.best_values.items():
            if param_name.startswith(pref):
                evalParms[param_name[len(pref):]] = value
        functions+=[hpg.hyperFinePredictionFreeAmps_pseudoVoigt(x=xInt,
                                                                **{kwarg:fittingkwargs[kwarg] for kwarg in evalKwargs},
                                                                **evalParms)]
        with open(f'{self.resultsPath}fitEvaluator/{pref}evalParms.pkl','wb') as file: pickle.dump(evalParms, file)
      return(functions)
    fitEvaluator(self.x)
    '''outputting results to files'''

    x_interp=np.linspace(np.min(self.x), np.max(self.x), 1000)
    for i,comp in enumerate(interpolator):
      prefix='bg' if i==0 else f'iso{i-1}'
      np.savetxt(f'{self.resultsPath}_{prefix}{self.suffix}.csv',np.c_[x_interp,comp(x_interp)],delimiter=',')
    self.plotFitResults(result, interpolator, **fittingkwargs)
    with open(f'{self.resultsPath}fitting_kwargs{self.suffix}.pkl','wb') as file: pickle.dump(fittingkwargs, file)
    with open(    f'{self.resultsPath}fit_params{self.suffix}.pkl','wb') as file: pickle.dump(result.params, file)
    with open(    f'{self.resultsPath}fit_result{self.suffix}.pkl','wb') as file: pickle.dump(result.params, file) #only doing this until I clean up line 249 in fullAnalysis
    with open(f'{self.resultsPath}fit_statistics{self.suffix}.pkl','wb') as file: pickle.dump(statsDic, file)
    # with open(f'{self.resultsPath}fit_interpolator{self.suffix}.pkl','wb') as file: pickle.dump(fitEvaluator, file)
    self.writePeakPositions(f'{self.resultsPath}peakPositions{self.suffix}.txt', result, **fittingkwargs)

    
  def loadFitResults(self):
    if os.path.exists(f'{self.resultsPath}fitting_kwargs{self.suffix}.pkl'):
      try:
        with open(f'{self.resultsPath}fit_params{self.suffix}.pkl','rb') as file: self.resultParams=pickle.load(file)
        with open(f'{self.resultsPath}fitting_kwargs{self.suffix}.pkl','rb') as file: self.fittingkwargs=pickle.load(file)
        with open(f'{self.resultsPath}fit_statistics{self.suffix}.pkl','rb') as file: self.fitStats=pickle.load(file)
      except Exception as e:
        print(f'Error: One or more fit result files not found in {self.resultsPath}. Ensure that you have run fitAndLogData() prior to loading fit results');
        print(e)
        quit()
      return(self.resultParams)
    else:
      return False #if there are no results saved to file

  def populateFrame(self, prefix='iso0',index=False):
    fixed_Aratio=self.fittingkwargs['fixed_Aratio'] if 'fixed_Aratio' in self.fittingkwargs.keys() else False
    spindex=int(prefix.lstrip('iso')); nuclearSpin=self.nuclearSpinList[spindex]
    aLower=self.resultParams[prefix+'_Alower'].value; aLower_uncertainty=self.resultParams[prefix+'_Alower'].stderr
    aUpper=self.resultParams[prefix+'_Aupper'].value; aUpper_uncertainty=self.resultParams[prefix+'_Aupper'].stderr
    aRatio=aLower/self.resultParams[prefix+'_Aupper'].value
    aRatio_uncertainty = (aRatio**2) *( (aLower_uncertainty/aLower)**2 + (aUpper_uncertainty/aUpper)**2)
    tempDict={
    "massNumber":round(self.mass),
    "mass":[self.mass],
    "mass_uncertainty":[self.mass_uncertainty if self.mass_uncertainty else 0],
    "I":[nuclearSpin],
    "aLower":[aLower],
    "aLower_uncertainty":[aLower_uncertainty],
    "aUpper":[aUpper],
    "aUpper_uncertainty":[aUpper_uncertainty],
    "aRatio":[aRatio],
    "aRatio_uncertainty":[0] if fixed_Aratio else [aRatio_uncertainty],
    "centroid":[self.resultParams[prefix+'_centroid'].value],
    "cent_uncertainty":[self.resultParams[prefix+'_centroid'].stderr],#,None,None,None,None]]
    "sigma":[self.resultParams[prefix+'_sigma'].value],
    "sigma_uncertainty":[self.resultParams[prefix+'_sigma'].stderr],
    "gamma":[self.resultParams[prefix+'_gamma'].value],
    "gamma_uncertainty":[self.resultParams[prefix+'_gamma'].stderr],
    }
    if self.energyCorrection:
      tempDict["uncorrectedCentroid"]=[self.uncorrectedSpectrum.resultParams[prefix+'_centroid'].value]
      tempDict["uncorrectedCentroid_uncertainty"]=[self.uncorrectedSpectrum.resultParams[prefix+'_centroid'].stderr]
    tempDict["avgScanTime"]=np.mean(self.spectrumFrame['avgScanTime'])-self.timeOffset
    if index: return(pd.DataFrame(tempDict, index=[index]))
    return(pd.DataFrame(tempDict))
  
  def __repr__(self):
    return(f'Spectrum object {hex(id(self))}, with attributes:\n{list(self.__dict__.keys())}')
    
if __name__ == '__main__':
  frequencyOffset = 1129900000
  scanTimeOffset=1716156655
  runsDictionary = {27:[16368,16369,16370,16389,16391,16392,16395,16396,16397,16410,16412,16413,16422,16424,16425,16426,
                        16429,16430,16439,16441,16442,16451,16458,16459,16470,16473,16474,16477,16492,16494,16495,16508,16510,16512]}
  jGround=0.5
  jExcited=0.5
  iNucDictionary={27:[2.5]}
  massDictionary = {27:26.981538408}
  
  mass_uncertaintyDictionary = {27:0.00000005}
  
  laserDictionary = {27:376.052850}
  for key in laserDictionary.keys(): laserDictionary[key]*=3E6

  timeStepDictionary= {27:[489,543]}

  tofDictionary={27: [23.45E-6,26.0E-6]}
  
  
  

  colinearity=False
  equal_fwhm=True
  cec_sim_toggle=False
  directoryPrefix='dummyTest'+'/equal_fwhm_'+str(equal_fwhm)+'/cec_sim_toggle_'+str(cec_sim_toggle!=False)

  def fittingFunction(x,y,yErr,**kwargs):
    result, interp = hpg.fitData(x, y, yErr, massDictionary[massNumber], iNucDictionary[massNumber], jGround, jExcited, **kwargs)
    return(result, interp)

  
  massNumber=27
  run=runsDictionary[massNumber][0]
  scanDirectory=str(massNumber)+'Al/'
  targetDirectory = 'Scan%d'%run
  tofWindow=tofDictionary[massNumber]
  # def fittingFunction(x,y,yErr,**kwargs):#passing this "specialized" function to spectrum constructor, to tell it how to fit
  #   result, interp = hpg.fitData(x, y, yErr, massDictionary[massNumber], iNucDictionary[massNumber], jGround, jExcited, **kwargs)
  #   return(result, interp)
  fittingFunction = lambda x, y, yErr, mass=massDictionary[massNumber],\
      iList=iNucDictionary[massNumber], jGround=jGround, jExcited=jExcited, **kwargs:\
        hpg.fitData(x, y, yErr, mass, iList, jGround, jExcited, **kwargs)
  
  spectrumKwargs={'runs':[run],'mass':massDictionary[massNumber],'mass_uncertainty':mass_uncertaintyDictionary[massNumber], 'jGround':jGround, 'jExcited':jExcited, 'nuclearSpinList':iNucDictionary[massNumber],
                  'laserFrequency':laserDictionary[massNumber],'colinearity':colinearity, 'directoryPrefix':directoryPrefix,'scanDirectory':scanDirectory, 'targetDirectory':targetDirectory, 'constructSpectrum':True,
                  'timeOffset':scanTimeOffset,'windowToF':tofDictionary[massNumber], 'cuttingColumn':'ToF','fittingFunction':fittingFunction}
  spectrum=Spectrum(energyCorrection=-8.392, **spectrumKwargs)
  print(spectrum)
  fittingkwargs={'transitionLabel':'P12-S12','cec_sim_data_path':cec_sim_toggle,'equal_fwhm':equal_fwhm, 'fixed_Aratio':False,'peakModel':'pseudoVoigt','spScaleable':False}
  result,_=spectrum.fitDat(**fittingkwargs)
  # spectrum.fitAndLogData(**fittingkwargs)
  # spectrum.loadFitResults()
  # test=spectrum.populateFrame()
  # print(test['centroid']-spectrum.frequencyOffset)
  print(spectrum)
  print(result.params['iso0_centroid'].value)#+spectrum.frequencyOffset)
  