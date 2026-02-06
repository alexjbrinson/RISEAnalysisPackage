import numpy as np
from sympy.physics.wigner import wigner_6j
from numba import njit
from RAP.SpectrumHandler import amu2eV, electronRestEnergy

def kNuc(iNuc, jElec, fTot):
  return(fTot*(fTot+1)-iNuc*(iNuc+1)-jElec*(jElec+1))

def racahCoefficients(iNuc, jElec1, fTot1, jElec2, fTot2):
  iNuc=float(iNuc);jElec1=float(jElec1); fTot1=float(fTot1); jElec2=float(jElec2); fTot2=float(fTot2)
  return((2*fTot1+1)*(2*fTot2+1)/(2*iNuc+1)*wigner_6j(jElec2,fTot2,iNuc,fTot1,jElec1,1)**2)

def energySplitting(A, B, iNuc, jElec, fTot):
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

def hfsLinesAndStrengths(iNuc,jElec1,jElec2,A1,A2,B1=0,B2=0):
  linesList=[]; strengthsList=[]
  f1List=np.arange(abs(iNuc-jElec1),iNuc+jElec1+1,1)
  f2List=np.arange(abs(iNuc-jElec2),iNuc+jElec2+1,1)
  for fTot1 in f1List:
    for fTot2 in f2List:
      if abs(fTot1-fTot2)<=1: #viable transition!
        shift1=energySplitting(A1, B1, iNuc, jElec1, fTot1)
        shift2=energySplitting(A2, B2, iNuc, jElec2, fTot2)
        linePos=shift2-shift1
        tStrength=racahCoefficients(iNuc, jElec1, fTot1, jElec2, fTot2)
        linesList+=[linePos];strengthsList+=[tStrength]
  return(linesList, strengthsList)

def propagateBeamEnergyCorrectionToCentroid(mass, centroid, laserFrequency, ΔEkin):
  #print('dummy check: mass = %.1f ; centroid = %.2f'%(mass, centroid))
  neutralRestEnergy=mass*amu2eV
  ionRestEnergy=neutralRestEnergy-electronRestEnergy
  #print(centroid, laserFrequency)
  #dcf_0=centroid;
  beta_0 = (centroid**2-laserFrequency**2)/(centroid**2+laserFrequency**2); gamma_0=1/np.sqrt(1-beta_0**2)
  voltage_0 = ionRestEnergy*(gamma_0-1)
  voltage_1 = voltage_0 + ΔEkin
  beta_1 = np.sqrt(1-((ionRestEnergy)/(voltage_1+ionRestEnergy))**2)
  dcf_1 = np.sqrt(1+beta_1)/np.sqrt(1-beta_1)*laserFrequency
  #print('test: voltage_0 = ',voltage_0)

  seriesExpansionScaling = centroid/((ionRestEnergy + voltage_0)*beta_0)
  #print('nu = %.5f+%.5f*ΔEkin'%(centroid, seriesExpansionScaling) )
  #print('total centroid shift = ', dcf_1-centroid)
  return(dcf_1)

def freqToVoltage(mass, laserFrequency, peakFreq, freqOffset=0):
  #this function takes a resonance frequency in MHz and converts it to eV
  neutralRestEnergy=mass*amu2eV
  ionRestEnergy=neutralRestEnergy-electronRestEnergy
  peakFreq+=freqOffset
  beta_0_amp = (peakFreq**2-laserFrequency**2)/(peakFreq**2+laserFrequency**2); #anti/collinearity doesn't matter, since I just square this to get gamma_0 anyway
  gamma_0=1/np.sqrt(1-beta_0_amp**2); #print(gamma_0)
  voltage_0 = ionRestEnergy*(gamma_0-1); #print(voltage_0)
  return(voltage_0)

def freqShiftToVoltageShift(mass, laserFrequency, peakFreq, peakShift, freqOffset=0):
  #this function takes a sidepeak fit result in MHz and converts it to eV
  voltage_0 = freqToVoltage(mass, laserFrequency, peakFreq, freqOffset=freqOffset)
  voltage_1 = freqToVoltage(mass, laserFrequency, peakFreq+peakShift, freqOffset=freqOffset)
  Δv = voltage_1-voltage_0
  return(Δv)

def voltageShiftToFrequencyShift(mass, laserFrequency, voltage0, voltageShift, freqOffset=0, colinearity=True, theta=0):
  neutralRestEnergy=mass*amu2eV
  ionRestEnergy=neutralRestEnergy-electronRestEnergy
  beta_0=np.sqrt(1-((ionRestEnergy)/(voltage0+ionRestEnergy))**2)
  beta_1=np.sqrt(1-((ionRestEnergy)/(voltage0+voltageShift+ionRestEnergy))**2)
  if colinearity: beta_0*=-1; beta_1*=-1
  #doppler corrected frequencies
  if False:#theta==0:
    dcf0 = np.sqrt(1+beta_0)/np.sqrt(1-beta_0)*laserFrequency
    dcf1 = np.sqrt(1+beta_1)/np.sqrt(1-beta_1)*laserFrequency
  else:
    dcf0 = (1+beta_0*np.cos(theta))/np.sqrt(1-beta_0**2)*laserFrequency
    dcf1 = (1+beta_1*np.cos(theta))/np.sqrt(1-beta_1**2)*laserFrequency
  Δf=dcf1-dcf0
  return(Δf)

def voltageToFrequency(mass, laserFrequency, voltage0, voltageShift, freqOffset=0, colinearity=True, theta=0):
  neutralRestEnergy=mass*amu2eV
  ionRestEnergy=neutralRestEnergy-electronRestEnergy
  beta_1=np.sqrt(1-((ionRestEnergy)/(voltage0+voltageShift+ionRestEnergy))**2)
  if colinearity: beta_0*=-1; beta_1*=-1
  #doppler corrected frequencies
  if False:#theta==0:
    dcf0 = np.sqrt(1+beta_0)/np.sqrt(1-beta_0)*laserFrequency
    dcf1 = np.sqrt(1+beta_1)/np.sqrt(1-beta_1)*laserFrequency
  else:
    dcf1 = (1+beta_1*np.cos(theta))/np.sqrt(1-beta_1**2)*laserFrequency
  return(dcf1)

@njit#(fastmath=True)
def generateSidePeaks(mass, laserFrequency, peakFreq, sp_fractions, cec_sim_list, frequencyOffset=0, colinearity=True):#, cecBinning=False):
  '''
  based on input mass, laser frequency, and resonance frequency, determine resonance in voltage space, then 
  apply e_losses from cec_sim_list, and finally transform back to frequency space to predict sidepeak positions
  '''
  originalPeakFreq=peakFreq#...this seems to change slightly if I don't do this
  mask = sp_fractions != 0 # Filtering non-zero elements
  cec_sim_list = cec_sim_list[mask]; sp_fractions = sp_fractions[mask]
  cec_sim_list[cec_sim_list<0] /=3 #update May 23, 2025: dividing energy loss channels by 3, for some reasons.

  neutralRestEnergy=mass*amu2eV
  ionRestEnergy=neutralRestEnergy-electronRestEnergy

  peakFreq+=frequencyOffset #need to use actual resonance frequency, not fitting frequency (which is offset), or else Doppler transforms are wrong
  beta_0_amp = (peakFreq**2-laserFrequency**2)/(peakFreq**2+laserFrequency**2); #anti/collinearity doesn't matter, since I just square this to get gamma_0 anyway 
  gamma_0=1/np.sqrt(1-beta_0_amp**2); #print(gamma_0)
  voltage_0 = ionRestEnergy*(gamma_0-1); #print(voltage_0)
  voltageList = voltage_0 - cec_sim_list; ##e-losses actually look like higher voltages, because you had to accelerate more to end up resonant after inelastic collision 
  betaList=np.sqrt(1-((ionRestEnergy)/(voltageList+ionRestEnergy))**2)
  if colinearity:
    dcfList = np.sqrt(1-betaList)/np.sqrt(1+betaList)*laserFrequency #doppler corrected frequencies
  else:
    dcfList = np.sqrt(1+betaList)/np.sqrt(1-betaList)*laserFrequency #doppler corrected frequencies
  dcfList-=frequencyOffset

  neg_mask = cec_sim_list < 0; pos_mask = cec_sim_list > 0
  sp_shifts=np.zeros(len(dcfList[neg_mask])+1)
  sp_shifts[1:]=dcfList[neg_mask]
  sp_shifts[0] = originalPeakFreq
  
  negativeOffsets=dcfList[pos_mask]-sp_shifts[0]
  negOffsetWeights=sp_fractions[pos_mask]

  #question: should I sum the positive e_losses linearly or in quadrature? TODO: Confirm that it makes sense to transform to MHz
  broadeningFactor = np.sum(negativeOffsets * negOffsetWeights) / np.sum(negOffsetWeights) if len(negOffsetWeights) > 0 else 0.0
  #Note: jit does not like my binning code, and honestly why bin if you can jit anyway? Keeping here in case I ever want to bring it back.
  # if cecBinning!=False: #if true, we are binning the simulated sidepeaks in bins of width cecBinning, and we need to calculate weighted means for sp_positions, and sum the appropriate fractiosn 
  #   bins=np.arange(sp_shifts.min(),sp_shifts.max()+cecBinning, cecBinning)#-cecBinning/2
  #   bindices = np.digitize(sp_shifts,bins)
  #   finalLength=len(np.unique(bindices))
  #   tempShifts=np.zeros(finalLength); tempFracts=np.zeros(finalLength)
  #   for i, bindex in enumerate(np.unique(bindices)):
  #     tempFracts[i] = np.sum(sp_fractions[bindices==bindex])
  #     tempShifts[i] = np.sum(sp_fractions[bindices==bindex]*sp_shifts[bindices==bindex])/tempFracts[i]
  #   sp_shifts=tempShifts
  #   sp_fractions=tempFracts
  broadeningList=np.zeros(len(sp_shifts)); broadeningList[0] = broadeningFactor
  sp_shifts-=sp_shifts[0]# I'm only returning the shifts to the main peak frequency to make the use case slightly more general
  return(sp_shifts, sp_fractions, broadeningList)

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