# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 09:50:05 2017

@author: artem
"""
import numpy as np
import pandas as pd
import gc
import matplotlib.pyplot as plt
import h5py
import csv
import os
import inspect
from time import sleep
import argparse
import glob
    

##import files from folder, return files list
def importfiles(path):  
    import os
    arr = []
    for dir, subdir, files in os.walk(path):
        for file in files:
            arr.append(os.path.join(dir, file))
    return arr
    
##return Signal as nparray 
def GetSignal(file):
    with h5py.File(file,'r') as hdf:
        Reads = hdf.get('/Raw/Reads/')
        Reads_items = list(Reads.items())
        Read = hdf.get('/Raw/Reads/'+Reads_items[0][0])
        Signal = np.array(Read.get('Signal'))
        return(Signal)
        
##plot a signal
def Signal_plot(Signal, start = 0, end = 5000):
    plt.plot(Signal[start:end])
    plt.title('Signal plot')
    plt.xlabel('index')
    plt.ylabel('V')
    return(plt.show())    
  
         
##get histogram of multiple signals:       
def Signal_len_hist(files_lst, start = 0, end = 5000):
    files_len = []
    for f in files_lst:
        Signal = GetSignal(f)
        Signal_len = Signal.shape[0]
        files_len.append(Signal_len)
    print (min(files_len))
    bins = np.arange(0, 300000 , 500)
    plt.xlim(start, end)
    plt.hist(files_len, bins=bins, alpha=0.5)
    plt.title('Signal length')
    plt.xlabel('variable X (bin size = 5)')
    plt.ylabel('Count')
    return(plt.show())

             
 ##finds Signal starting point       
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
            if np.var(Signal[Start_point[0]+50:Start_point[0]+80]) > np.var(Signal[Start_point[0]-80:Start_point[0]-50]):
                return(Start_point[0])
    ## if couldnt find valid start poin then return 0 as start point
    return int("0")
        
            




import random

def main():

    pathOfFast5Folder = args.pathOfFast5Files
    prefixFast5Folder = args.prefixName
    ##get random normalized signals from a tar file.
    ZeroIndexCounter = 0
    MoreThan1000AfterCleaning = 0

    allSignallArray = []
    fromStartSignal = []
    fromStartOnly1KB = []
    AVG_fromStartOnly1KB = []
    Normed_fromStartOnly1KB = []
    fromStartSignal_minor = []
    fromStartOnly1KB_minor = []
    AVG_fromStartOnly1KB_minor = []
    Normed_fromStartOnly1KB_minor = []
    NormalizedSig = []
    Rfiles = []
    filename = inspect.getframeinfo(inspect.currentframe()).filename
    tar_dir_path = pathOfFast5Folder
    print("working on tar file: ",prefixFast5Folder)
    ZeroIndexCounter = 0
    MoreThan1000AfterCleaning = 0
    MoreThan10000AfterCleaning = 0
    fromStartSignal = []
    allSignallArray = []
    fromStartOnly10KB = []
    AVG_fromStartOnly10KB = []
    Normed_fromStartOnly10KB = []
    fromStartOnly1KB = []
    AVG_fromStartOnly1KB = []
    Normed_fromStartOnly1KB = []

    total_num_files = 65000
    major_num_files = 60000
    test_num_files = 10000
    Rfiles = glob.glob(pathOfFast5Folder +'*.fast5')
    print(pathOfFast5Folder +'*.fast5')
    test_num_files = int(len(Rfiles)/10)


    Rfiles1 = Rfiles[0:int(-1*test_num_files)]  
    Rfiles2 = Rfiles[int(-1*test_num_files):]
    print("\nMajor signal list", prefixFast5Folder, str(len(Rfiles1)))
    fileCounter = 0
    for file_name in Rfiles1:
        sleepCounter = 0
        fileCounter += 1
        if fileCounter %1000 == 0:
            print (fileCounter)
        pathOfCurrentRead = os.path.join(pathOfFast5Folder,file_name)
        try:
            currentSig = GetSignal(file_name)
        except:
            
            ZeroIndexCounter += 1
            continue
        currentSigLen = len(currentSig)
        currentSigStart = Signalstart(currentSig)
        if currentSigStart == 0:
            ZeroIndexCounter += 1
            continue
        fromStartSignal.append(currentSig[currentSigStart:])
        allSignallArray.append(currentSig)
        if currentSigLen-currentSigStart > 1000:
            MoreThan1000AfterCleaning += 1
            sig_1kb = currentSig[currentSigStart:currentSigStart+1000]
            fromStartOnly1KB.append(sig_1kb)
        if currentSigLen-currentSigStart > 5000:
            MoreThan10000AfterCleaning += 1
            sig_10kb = currentSig[currentSigStart:currentSigStart+5000]
            fromStartOnly10KB.append(sig_10kb)
            del  sig_10kb
            del  sig_1kb
    print ("ZeroindexCounter for major file: ", ZeroIndexCounter)
    print ("more than 1k after cleaning: ", MoreThan1000AfterCleaning)
    print ("more than 10k after cleaning: ", MoreThan10000AfterCleaning)
    print("printing the length statistics (all signals length)")
    print(sum( map(len, allSignallArray) ) / len(allSignallArray))
    csvFileBaseName = prefixFast5Folder
    csvFile10KbName = csvFileBaseName + "5KbTrain.csv"
    del Rfiles1 
    gc.collect()
    np.savetxt("separating_files/"+prefixFast5Folder+"/"+csvFile10KbName,\
            fromStartOnly10KB, fmt='%1i', delimiter=",")
    
    print("\nMinor signal list", prefixFast5Folder, str(len(Rfiles2)))
    ZeroIndexCounter = 0
    MoreThan1000AfterCleaning = 0
    MoreThan10000AfterCleaning = 0
    fromStartSignal = []
    allSignallArray = []
    fromStartOnly10KB = []
    AVG_fromStartOnly10KB = []
    Normed_fromStartOnly10KB = []
    fromStartOnly1KB = []
    AVG_fromStartOnly1KB = []
    Normed_fromStartOnly1KB = []
    for file_name in Rfiles2:
        fileCounter += 1
        pathOfCurrentRead = os.path.join(pathOfFast5Folder,file_name)
        try:
            currentSig = GetSignal(file_name)
        except:
            ZeroIndexCounter += 1
            continue
        currentSigLen = len(currentSig)
        currentSigStart = Signalstart(currentSig)
        if currentSigStart == 0:
            ZeroIndexCounter += 1
            continue
        fromStartSignal.append(currentSig[currentSigStart:])
        allSignallArray.append(currentSig)
        if currentSigLen-currentSigStart > 1000:
            MoreThan1000AfterCleaning += 1
            sig_1kb = currentSig[currentSigStart:currentSigStart+1000]
            fromStartOnly1KB.append(sig_1kb )
        if currentSigLen-currentSigStart > 5000:
            MoreThan10000AfterCleaning += 1
            sig_10kb = currentSig[currentSigStart:currentSigStart+5000]
            fromStartOnly10KB.append(sig_10kb )
        sig_10kb = None
        sig_1kb = None
        AVG_signal = None
        sig_10kb_norm = None
    print ("ZeroindexCounter for minor file: ", ZeroIndexCounter)
    print ("more than 1k after cleaning: ", MoreThan1000AfterCleaning)
    print ("more than 10k after cleaning: ", MoreThan10000AfterCleaning)
    print("printing the length statistics (all signals length)")
    print(sum( map(len, allSignallArray) ) / len(allSignallArray))
    csvFileBaseName = prefixFast5Folder
    csvFile10KbName = csvFileBaseName + "5KbTest.csv"
    np.savetxt("separating_files/"+prefixFast5Folder+"/"+csvFile10KbName,\
            fromStartOnly10KB, fmt='%1i', delimiter=",")
    
    







if __name__ == '__main__':
    parser = argparse.ArgumentParser('Get CSV file with the signals from all fast5 files in the specified folder')
    parser.add_argument('--pathOfFast5Files', required=True,
                        help='path to the folder with all wanted fast5 files')
    parser.add_argument('--prefixName', required=True,
                        help='name of the folder to save the csv file in')
    args = parser.parse_args()
    main()
