from setuptools import setup

with open("README.md") as f:
    long_description = f.read()

with open("requirements.txt") as f:
    requirements = f.read().splitlines()
    
setup(name="modds",
      version="0.1",
      description="Bayesian modeling of DM halos",
      long_description=long_description,
      author="Asher Wasserman",
      author_email="adwasser@ucsc.edu",
      url="https://github.com/adwasser/modds",
      packages=["modds"],
      scripts=["bin/modds"],
      install_requires=requirements,
      classifiers=[
          "Intended Audience :: Science/Research",
          "License :: OSI Approved :: MIT License",
          "Natural Language :: English",
          "Operating System :: OS Independent",
          "Programming Language :: Python :: 3 :: Only",
          "Topic :: Scientific/Engineering :: Astronomy"
      ]
)
