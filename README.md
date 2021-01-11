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

Copy the files from this repository to the folder where `simple.py` is located. Add these lines to load pytorch and to load the model in the beginning of `simple.py`:

```
import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets, transforms
from nanopore_dataloader import NanoporeDataset, differences_transform, noise_transform,\
								startMove_transform, cutToWindows_transform, startMove_transform_test

model_path = "Models/13Jul_bnLSTM_32win_512Hidden_1layer_winlen32_withDropout_outputLastStep/Nanopore_model.pth"
model= torch.load(model_path)

def Signalstart(Signal):
    Start_point, Pro_start, Pre_start = 0, [], []
    Signal_lst = Signal.tolist()
    Start_point = (Signal_lst.index(max(Signal_lst[10:3000])),max(Signal_lst[10:3000]))
    Pre_start = Signal_lst[Start_point[0]-19:Start_point[0]-1]
    Pro_start = Signal_lst[Start_point[0]+1:Start_point[0]+19]
    if not Pro_start or not Pre_start:
        return int("0")
    if Start_point[1] > sum(Pre_start)/len(Pre_start):
        if Start_point[1] > sum(Pro_start)/len(Pro_start):
            if numpy.var(Signal[Start_point[0]+50:Start_point[0]+80]) > numpy.var(Signal[Start_point[0]-80:Start_point[0]-50]):
                return(Start_point[0])
    ## if couldnt find valid start poin then return 0 as start point
    return int("0")
```

Then, after line 94 `time.sleep(delay)` add:

```
badMitoReadCounter = 0
notMitoReadCounter = 0
yesMitoReadCounter = 0
```

After line 99 `read_batch = client.get_read_chunks(batch_size=batch_size, last=True)` add:

```
readsToAnalyzeList = []
numpy_sample_List = []
```

Replace the block of code in the for loop in lines 100-113 starting with `for channel, read in read_batch:` and ending with `t1 = time.time()` with this code:

```
    # convert the read data into a numpy array of correct type
    raw_data = numpy.fromstring(read.raw_data, "int16")
    stride = 1
    winLength = 1
    seqLength = 2000
    raw_data = raw_data.astype("int16")
    signalStart = Signalstart(raw_data)
    if (signalStart == 0):
        badMitoReadCounter += 1
        client.unblock_read(channel, read.number)
        continue
    raw_data=raw_data[signalStart:]
    if (len(raw_data) < 2001):
        badMitoReadCounter += 1
        client.unblock_read(channel, read.number)
        continue
    readsToAnalyzeList.append((channel, read))
    raw_data=differences_transform(raw_data)
    raw_data=cutToWindows_transform(raw_data, seqLength, stride, winLength)
    numpy_sample_List.append(raw_data)

if len(numpy_sample_List) == 0:
    pass
else:
    numpy_sample_npList = numpy.stack(numpy_sample_List)

    tensorRead = torch.from_numpy(numpy_sample_npList).float()
    tensorRead = Variable(tensorRead).cuda()
    logits = model(input_=tensorRead)
    for channel_read_index, channel_read_tupple in enumerate(readsToAnalyzeList):
        channel = channel_read_tupple[0]
        read = channel_read_tupple[1]
        if logits[channel_read_index][1].data > 0.999:
            yesMitoReadCounter += 1
            client.stop_receiving_read(channel, read.number)
            print("YESS")
        else:
            client.unblock_read(channel, read.number)
            notMitoReadCounter += 1
```

To print the statistics at the end, add these lines after line 117  `logger.info('Finished analysis of reads as client stopped.')`:

```   
print(badMitoReadCounter)
print(notMitoReadCounter)
print(yesMitoReadCounter)
```

This code should perform selective sequencing for mitochondrial DNA using the pretrained model provided in the `Models` folder.







