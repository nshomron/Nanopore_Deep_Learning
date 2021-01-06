# Nanopore real time selective sequencing using deep learning
A repository for Real-Time classification on Nanopore sequencing raw signal using deep-learning methods.

This is the official implamentation of the code in "Real-time selective sequencing using nanopores and deep learning"

## Environment & requirments
The code was tested on:

* Linux 4.15.0-128-generic
* 16.04.1-Ubuntu
* Python 3.6.4
* (A GPU with cuda support is recommended but not required)


In order to train and test deep learning models the requirments are:
* torch==0.4.1
* torchvision==0.8.2
* matplotlib
* pandas
* numpy
* scikit_learn


## Basic usage
To test the installation simply run the next command:

    python nanopore_training.py --hidden-size 512 --batch-size 32 --max-iter 100 --gpu

(remove the `--gpu` tag to run on CPU instead of GPU)

This command will run a simple training routine on the exemplary files located in the Data folder.

Currently the deep learning scripts set up to classify between two categories of signals: "Mito" signals (nanopore raw signal data of molecules that were mapped to the mitochondria) and "non-Mito" signals (nanopore raw signal from molecules that were mapped to the rest of the genome). The example data is seperated into the two classes and there is another separation for training data and testing data (validation)

## User generated data

In order to generate the required files for the deep learning models from fast5 files (nanopore sequencing output files) follow the steps in the Preparing_fast5_for_analysis folder.




