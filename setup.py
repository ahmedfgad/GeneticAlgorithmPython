import setuptools  

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(  
    name="pygad",  
    version="2.14.3",  
    author="Ahmed Fawzy Gad",
    install_requires=["numpy", "matplotlib",],
    author_email="ahmed.f.gad@gmail.com",  
    description="PyGAD: A Python 3 Library for Building the Genetic Algorithm and Training Machine Learning Algoithms (Keras & PyTorch).",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ahmedfgad/GeneticAlgorithmPython",
    license="BSD-3-Clause",
    classifiers=[
        "License :: OSI Approved :: BSD License"
    ],
    packages=setuptools.find_packages())
