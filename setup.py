from setuptools import setup#, find_packages

setup(
  name='RISEAnalysisPackage',
  version='0.1',
  py_modules=['hyperfinePredictorGREAT', 'dataMunger', 'BeamEnergyAnalysis'],
  install_requires=['numpy', 'pandas', 'lmfit', 'matplotlib','sympy', 'scipy']
)