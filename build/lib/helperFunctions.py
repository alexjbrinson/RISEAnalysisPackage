import numpy as np
from sympy.physics.wigner import wigner_6j
from spectrumHandler import amu2eV, electronRestEnergy

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

def propagateBeamEnergyCorrectionToCentroid(mass, centroid, laserFreq, ΔEkin):
  #print('dummy check: mass = %.1f ; centroid = %.2f'%(mass, centroid))
  neutralRestEnergy=mass*amu2eV
  ionRestEnergy=neutralRestEnergy-electronRestEnergy
  #print(centroid, laserFreq)
  #dcf_0=centroid;
  beta_0 = (centroid**2-laserFreq**2)/(centroid**2+laserFreq**2); gamma_0=1/np.sqrt(1-beta_0**2)
  voltage_0 = ionRestEnergy*(gamma_0-1)
  voltage_1 = voltage_0 + ΔEkin
  beta_1 = np.sqrt(1-((ionRestEnergy)/(voltage_1+ionRestEnergy))**2)
  dcf_1 = np.sqrt(1+beta_1)/np.sqrt(1-beta_1)*laserFreq
  #print('test: voltage_0 = ',voltage_0)

  seriesExpansionScaling = centroid/((ionRestEnergy + voltage_0)*beta_0)
  #print('nu = %.5f+%.5f*ΔEkin'%(centroid, seriesExpansionScaling) )
  #print('total centroid shift = ', dcf_1-centroid)
  return(dcf_1)

def freqToVoltage(mass, laserFreq, peakFreq, freqOffset=0):
  #this function takes a resonance frequency in MHz and converts it to eV
  neutralRestEnergy=mass*amu2eV
  ionRestEnergy=neutralRestEnergy-electronRestEnergy
  peakFreq+=freqOffset
  beta_0_amp = (peakFreq**2-laserFreq**2)/(peakFreq**2+laserFreq**2); #anti/collinearity doesn't matter, since I just square this to get gamma_0 anyway
  gamma_0=1/np.sqrt(1-beta_0_amp**2); #print(gamma_0)
  voltage_0 = ionRestEnergy*(gamma_0-1); #print(voltage_0)
  return(voltage_0)

def freqShiftToVoltageShift(mass, laserFreq, peakFreq, peakShift, freqOffset=0):
  #this function takes a sidepeak fit result in MHz and converts it to eV
  voltage_0 = freqToVoltage(mass, laserFreq, peakFreq, freqOffset=freqOffset)
  voltage_1 = freqToVoltage(mass, laserFreq, peakFreq+peakShift, freqOffset=freqOffset)
  Δv = voltage_1-voltage_0
  return(Δv)

def voltageShiftToFrequencyShift(mass, laserFreq, voltage0, voltageShift, freqOffset=0, colinearity=True, theta=0):
  neutralRestEnergy=mass*amu2eV
  ionRestEnergy=neutralRestEnergy-electronRestEnergy
  beta_0=np.sqrt(1-((ionRestEnergy)/(voltage0+ionRestEnergy))**2)
  beta_1=np.sqrt(1-((ionRestEnergy)/(voltage0+voltageShift+ionRestEnergy))**2)
  if colinearity: beta_0*=-1; beta_1*=-1
  #doppler corrected frequencies
  if False:#theta==0:
    dcf0 = np.sqrt(1+beta_0)/np.sqrt(1-beta_0)*laserFreq
    dcf1 = np.sqrt(1+beta_1)/np.sqrt(1-beta_1)*laserFreq
  else:
    dcf0 = (1+beta_0*np.cos(theta))/np.sqrt(1-beta_0**2)*laserFreq
    dcf1 = (1+beta_1*np.cos(theta))/np.sqrt(1-beta_1**2)*laserFreq
  Δf=dcf1-dcf0
  return(Δf)

def voltageToFrequency(mass, laserFreq, voltage0, voltageShift, freqOffset=0, colinearity=True, theta=0):
  neutralRestEnergy=mass*amu2eV
  ionRestEnergy=neutralRestEnergy-electronRestEnergy
  beta_1=np.sqrt(1-((ionRestEnergy)/(voltage0+voltageShift+ionRestEnergy))**2)
  if colinearity: beta_0*=-1; beta_1*=-1
  #doppler corrected frequencies
  if False:#theta==0:
    dcf0 = np.sqrt(1+beta_0)/np.sqrt(1-beta_0)*laserFreq
    dcf1 = np.sqrt(1+beta_1)/np.sqrt(1-beta_1)*laserFreq
  else:
    dcf1 = (1+beta_1*np.cos(theta))/np.sqrt(1-beta_1**2)*laserFreq
  return(dcf1)