from setuptools import setup#, find_packages

setup(
  name='RISEAnalysisPackage',
  version='0.1',
  py_modules=['dataMunger', 'spectrumHandler','hyperfinePredictorGREAT', 'BeamEnergyAnalysis','onlineAnalysis', 'SpectrumClass'],
  install_requires=['numpy','sympy', 'scipy', 'pandas','polars','matplotlib','lmfit','numba']
)