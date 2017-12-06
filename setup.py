from setuptools import setup

__version__ = '0.2.0'

setup(
    name='pyfit',
    version=__version__,
    author='Eshin Jolly',
    author_email='eshin.jolly.gr@dartmouth.edu',
    install_requires=[
    'numpy>=1.9',
    'seaborn>=0.7.0',
    'lmfit==0.9.7',
    'matplotlib>=2.0.2',
    'scipy>=0.19.1',
    'pandas>=0.19.0'
    ],
    license='LICENSE.txt',
    description='A Python wrapper package to fit computational models.',
    keywords = ['modeling', 'statistics', 'analysis','optimization','minimization'],
    packages=['pyfit'],
    classifiers = [
        "Programming Language :: Python",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License"
    ]
)
