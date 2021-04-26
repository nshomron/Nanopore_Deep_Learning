# Nanopore real time selective sequencing using deep learning
A repository for Real-Time classification on Nanopore sequencing raw signal using deep-learning methods.

This is the official implementation of the code in "Real-time selective sequencing using nanopores and deep learning"

## Environment & requirements
The code was tested on:

* Linux 4.15.0-128-generic
* 16.04.1-Ubuntu
* Python 3.6.4
* (A GPU with cuda support is recommended but not required)


In order to train and test deep learning models the requirements are:
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

Result - The script will load the example data (4 files, mitochondrial data for training, non-mitochondrial data for training , mitochondrial data for testing, non-mitochondrial data for testing) and start the training process. The model is initiated as `bnLSTM_32window` model, or one can load a previously trained model by uncommenting line #215 (`# model= torch.load(saved_model_path)`) in the `nanopore_training.py` file. During training the progress will be printed to the console, and the progress plots will be saved in `results_onHek` folder.

## User generated data

In order to generate the required files for the deep learning models from fast5 files (nanopore sequencing output files) follow the steps in the Preparing_fast5_for_analysis folder.

## Real-time selective sequencing

The API for real-time application for nanopore sequencing (called "Read Until") is a proprietary code, it is possible to install the software following the official instructions at https://github.com/nanoporetech/read_until_api .

The API comes with it's own python interpreter, In order to use our model in conjunction with the read until feature it is required to install pytorch for this interpreter as well. (this process  might change in future read until api versions). Installing external python libraries like pytorch could be done like so:

```
sudo /opt/ONT/MinKNOW/ont-python/bin/python -m pip install http://download.pytorch.org/whl/cpu/torch-0.4.0-cp27-cp27m-linux_x86_64.whl
sudo /opt/ONT/MinKNOW/ont-python/bin/python -m pip install torchvision 
```

After installing pytorch, the simplest way to test selective sequencing using deep learning is to use the `simple.py` script.

Use this repository to install the Read Until API with a modified "simple.py" file for running selective sequencing with our model: https://github.com/nshomron/read_until_api
The simple file is located here: https://github.com/nshomron/read_until_api/blob/release/read_until/simple.py

This code should perform selective sequencing for mitochondrial DNA using the pretrained model provided in the `Models` folder.







