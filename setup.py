import setuptools


with open("README.md", "r") as fh:
  long_description = fh.read()

setuptools.setup(
  name="ensemble-pkg",
  version="0.0.3",
  author="sarthfrey",
  author_email="sarth.frey@gmail.com",
  description="Combine models, easily.",
  long_description=long_description,
  long_description_content_type="text/markdown",
  url="https://github.com/sarthfrey/ensemble",
  packages=setuptools.find_packages(),
  classifiers=[
    "Development Status :: 1 - Planning",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3",
    "Natural Language :: English",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Software Development :: Libraries :: Python Modules",
    "Topic :: Software Development :: Version Control :: Git",
    "Typing :: Typed",
  ],
  install_requires=[
    "numpy>=1.0.0",
  ],
)
