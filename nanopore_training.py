"""Train the model using MNIST dataset."""
import argparse
import os
from datetime import datetime
import time
from functools import partial
import pandas as pd
import numpy as np
import inspect
from sklearn.utils import shuffle
import math
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import csv
import itertools
from shutil import copyfile

import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
import re
from nanopore_dataloader import NanoporeDataset, differences_transform, noise_transform,\
								startMove_transform, cutToWindows_transform, RangeNormalize, \
								ReduceLROnPlateau
from nanopore_models import bnLSTM, bnLSTM_32window, regGru_32window_hidden_BN,\
							simpleCNN_10Layers_noDilation_largeKernel_withDropout,\
							VDCNN_gru_1window_lastStep



# torch.multiprocessing.set_start_method('spawn')  # Fix for cuda multiprocessing errors


### Command example to run DL training
### python nanopore_training.py --hidden-size 512 --batch-size 32 --max-iter 100 --gpu

def adjust_learning_rate(optimizer, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def main():
	## setting parameters
	hidden_size = args.hidden_size
	batch_size = args.batch_size
	max_iter = args.max_iter
	use_gpu = args.gpu
	filesToLoadFrom = args.gpu
	shuffleDatasets = True
	linesToLoad_chrom = 10000
	linesToLoad_mito = 10000
	# batchesForTest - number of times test dataset is reshuffled and rechecked
	batchesForTest = 3


	stride = 1
	winLength = 1
	seqLength = 2000
	outChannele = 64



	## Loading training data and setting logging location
	torch.manual_seed(7)
	folderWithDataFiles = "./Data/Hek1_example_data/"
	currentModelFolder = "results_onHek"
	searchModelsInFolder = "Models/"
	if not os.path.exists(currentModelFolder):
		os.makedirs(currentModelFolder)
	
	pattern_chrom = re.compile("FAF13387-Hek_run1.minimap2.not-Mito5KbTrain\.csv")
	pattern_Mito = re.compile("FAF13387-Hek_run1.minimap2.Mito5KbTrain\.csv")

	trainDataList_Chrom = []
	trainDataList_Mito = []
	for file in os.listdir(folderWithDataFiles):
		if pattern_chrom.match(file):
			print("Loading: ", file)
			trainDataList_Chrom.append(pd.read_csv(folderWithDataFiles+file,header = None, nrows=linesToLoad_chrom, dtype = np.int16))
		if pattern_Mito.match(file):
			print("Loading: ", file)
			trainDataList_Mito.append(pd.read_csv(folderWithDataFiles+file,header = None, nrows=linesToLoad_mito, dtype = np.int16))
	wholeData_Chrom = pd.concat(trainDataList_Chrom).values
	wholeData_Mito = pd.concat(trainDataList_Mito).values
	del trainDataList_Chrom
	del trainDataList_Mito

	
	pattern_chrom_TEST = re.compile("FAF13387-Hek_run1.minimap2.not-Mito5KbTest\.csv")
	pattern_Mito_TEST = re.compile("FAF13387-Hek_run1.minimap2.Mito5KbTest\.csv")


	trainDataList_Chrom_TEST = []
	trainDataList_Mito_TEST = []
	for file in os.listdir(folderWithDataFiles):
		if pattern_chrom_TEST.match(file):
			print("Loading: ", file)
			trainDataList_Chrom_TEST.append(pd.read_csv(folderWithDataFiles+file,header = None, nrows=linesToLoad_chrom, dtype = np.int16))
		if pattern_Mito_TEST.match(file):
			print("Loading: ", file)
			trainDataList_Mito_TEST.append(pd.read_csv(folderWithDataFiles+file,header = None, nrows=linesToLoad_mito, dtype = np.int16))
	wholeData_Chrom_TEST = pd.concat(trainDataList_Chrom_TEST).values
	wholeData_Mito_TEST = pd.concat(trainDataList_Mito_TEST).values
	del trainDataList_Chrom_TEST
	del trainDataList_Mito_TEST


	print("Size of the training Chrom and Mito:")
	print(len(wholeData_Chrom))
	print(len(wholeData_Mito))
	print("Size of the testing Chrom and Mito:")
	print(len(wholeData_Chrom_TEST))
	print(len(wholeData_Mito_TEST))

	## creating Labels

	wholeData_Chrom_labels = np.full(len(wholeData_Chrom), 0, dtype="int16")
	wholeData_Mito_labels = np.full(len(wholeData_Mito), 1, dtype="int16")

	wholeData_Chrom_labels_TEST = np.full(len(wholeData_Chrom_TEST), 0, dtype="int16")
	wholeData_Mito_labels_TEST = np.full(len(wholeData_Mito_TEST), 1, dtype="int16")

	## creating Datasets

	transform = transforms.Compose([
					transforms.Lambda(lambda x: startMove_transform(x)),
					transforms.Lambda(lambda x: differences_transform(x)),
					transforms.Lambda(lambda x: cutToWindows_transform(x, seqLength, stride, winLength)),
					transforms.Lambda(lambda x: noise_transform(x)),
				])

	train_chrom_dataset = NanoporeDataset(torch.from_numpy(wholeData_Chrom),
							torch.from_numpy(wholeData_Chrom_labels), shuffle = shuffleDatasets, transform = transform)
	train_mito_dataset = NanoporeDataset(torch.from_numpy(wholeData_Mito),
							torch.from_numpy(wholeData_Mito_labels), shuffle = shuffleDatasets, transform = transform)
	test_chrom_dataset = NanoporeDataset(torch.from_numpy(wholeData_Chrom_TEST),
							torch.from_numpy(wholeData_Chrom_labels_TEST), shuffle = shuffleDatasets, transform = transform)
	test_mito_dataset = NanoporeDataset(torch.from_numpy(wholeData_Mito_TEST),
							torch.from_numpy(wholeData_Mito_labels_TEST), shuffle = shuffleDatasets, transform = transform)


	summaryFile = open(str(currentModelFolder)+'/summaryFile_Cliveome.csv', 'w')
	summaryFile.write('Name, Mito_Total_Reads, Mito_Correct_Reads, Chrom_Total_Reads, Chrom_Correct_Reads \n')
	modelFolderPattern = re.compile(".*_winlen.*")
	for currentTestingModelFolder in os.listdir(searchModelsInFolder):
		if modelFolderPattern.match(currentTestingModelFolder):

			newFinetuneFolder = currentModelFolder + "/" + currentTestingModelFolder
			if not os.path.exists(newFinetuneFolder):
				os.makedirs(newFinetuneFolder)


			print("working folder is: ", currentTestingModelFolder)
			splitModelFolder = currentTestingModelFolder.split("_")
			for splitWord in splitModelFolder:
				print(splitWord)
				if "winlen" in splitWord:
					currentWinLen = int(splitWord.replace("winlen", ""))
					if currentWinLen == 1:
						stride = 1
						winLength = 1
						seqLength = 2000
					elif currentWinLen == 32:
						stride = 10
						winLength = 32
						seqLength = 200




			num_of_workers = 0
			use_pin_memmory = False
			train_chrom_dataloader = iter(DataLoader(dataset=train_chrom_dataset,
													   batch_size=batch_size, 
													   shuffle=shuffleDatasets,
													   drop_last = False,
													   num_workers = num_of_workers, pin_memory = use_pin_memmory))
			train_mito_dataloader = iter(DataLoader(dataset=train_mito_dataset,
													   batch_size=batch_size, 
													   shuffle=shuffleDatasets,
													   drop_last = False,
													   num_workers = num_of_workers, pin_memory = use_pin_memmory))
			test_chrom_dataloader = iter(DataLoader(dataset=test_chrom_dataset,
													   batch_size=batch_size, 
													   shuffle=shuffleDatasets,
													   drop_last = False,
													   num_workers = num_of_workers, pin_memory = use_pin_memmory))
			# test_mito_dataloader = iter(torch.utils.data.DataLoader(dataset=test_mito_dataset,
			test_mito_dataloader = iter(DataLoader(dataset=test_mito_dataset,
													   batch_size=batch_size, 
													   shuffle=shuffleDatasets,
													   drop_last = False,
													   num_workers = num_of_workers, pin_memory = use_pin_memmory))



			## Load any model from "nanopore_models" and set the parameters as in the example below

			# model = VDCNN_withDropout_normalMaxPool(input_size=winLength, hidden_size=hidden_size,\
			#  max_length=seqLength, n_classes=2, depth=9,\
			#  n_fc_neurons=1024, shortcut=True,\
			#  dropout=0.5)
			model = bnLSTM_32window(input_size=winLength, hidden_size=hidden_size, max_length=seqLength,  num_layers=1,
				 use_bias=True, batch_first=True, dropout=0.5,num_classes = 2)
			

			saved_model_path = '{}/{}/{}'.format(searchModelsInFolder,currentTestingModelFolder, "Nanopore_model.pth")
			
			## or uncomment to load previously saved model located in saved_model_path 

			# model= torch.load(saved_model_path)

			if use_gpu:
				torch.set_default_tensor_type('torch.cuda.FloatTensor')
				model.cuda()

			optim_lr = 0.001
			loss_fn = nn.CrossEntropyLoss().cuda()
			
			kappa = 0.01
			momentum = 0.5
			optimizer = torch.optim.Adam(model.parameters(), lr=optim_lr, eps=1e-05 ,weight_decay=0)
			scheduler = ReduceLROnPlateau(optimizer, 'min')


			## setting arrays for logging accuracies/losses
			lossArray = []
			accArray = []
			### lossArryForTest is required in order to plot one graph of train vs valid loss
			lossArryForTest = []
			accArryForTest = []
			lossArrayTest = []
			accArrayTest = []
			allChromAcc = []
			allChromAccTest = []
			minLoss = 999
			maxtestAcc= 0
			minLossModel = None
			matplotlib.interactive(False)
			lossPlt = plt.figure()


			def compute_loss_accuracy(data, label,model = model ,validation = False):
				if validation:
					logits = model(input_=data)
					loss = loss_fn(input=logits, target=label)
					accuracy = (logits.max(1)[1] == label).float().mean()
					return loss, accuracy, logits

				h_n = model(input_=data)
				h_n2 = model(input_=data)
				logits = h_n
				## fraternal loss
				kappa_logits = h_n2
				loss = 1/2*(loss_fn(logits, label) + loss_fn(kappa_logits, label))
				loss = loss + kappa * (logits - kappa_logits).pow(2).mean()
				accuracy = (logits.max(1)[1] == label).float().mean()
				return loss, accuracy, logits
			

			## setting variables to compute timings
			WholeTime = 0
			Stage1Time = 0
			Stage22Time = 0
			Stage23Time = 0
			Stage24Time = 0
			Stage2Time = 0
			Stage3Time = 0
			Stage4Time = 0
			Stage5Time = 0
			Stage6Time = 0
			Stage7Time = 0
			totalTime = 0
			currentEpochMito = 0
			currentEpochChrom = 0
			currentBatchNumber = 0
			mito_StopIter = False
			chrom_StopIter = False
			numTotalMitoSamples = 0
			numTotalChromSamples = 0
			numCorrectMitoSamples = 0
			numCorrectChromSamples = 0


			while currentEpochMito < max_iter:
				startTime = time.time()
				currentBatchNumber += 1
				stage1 = time.time()
				try:
					currentBatchChrom, currentBatchChrom_labels = train_chrom_dataloader.__next__()
				except StopIteration:
					print("Mito Is in Epoch: ",str(currentEpochMito))
					print("Chrom Is in Epoch: ",str(currentEpochChrom))
					currentEpochChrom += 1
					train_chrom_dataset = NanoporeDataset(torch.from_numpy(wholeData_Chrom),
											torch.from_numpy(wholeData_Chrom_labels),
											shuffle = shuffleDatasets, transform = transform)

					train_chrom_dataloader = iter(DataLoader(dataset=train_chrom_dataset,
															   batch_size=batch_size, 
															   shuffle=shuffleDatasets,
															   drop_last = False,
															   num_workers = num_of_workers, pin_memory = use_pin_memmory))
				
					currentBatchChrom, currentBatchChrom_labels = train_chrom_dataloader.__next__()
				try:
					currentBatchMito, currentBatchMito_labels = train_mito_dataloader.__next__()
				except StopIteration:
					currentEpochMito += 1
					print("Mito Is in Epoch: ",str(currentEpochMito))
					print("Chrom Is in Epoch: ",str(currentEpochChrom))
							
					train_mito_dataset = NanoporeDataset(torch.from_numpy(wholeData_Mito),
											torch.from_numpy(wholeData_Mito_labels),
											shuffle = shuffleDatasets, transform = transform)

					train_mito_dataloader = iter(DataLoader(dataset=train_mito_dataset,
															   batch_size=batch_size, 
															   shuffle=shuffleDatasets,
															   drop_last = False,
															   num_workers = num_of_workers, pin_memory = use_pin_memmory))
					currentBatchMito, currentBatchMito_labels = train_mito_dataloader.__next__()

				stage2 = time.time()

				## checking if needs to start new epoch based on two separate datasets

				if chrom_StopIter and mito_StopIter	:
					break
				elif not chrom_StopIter and mito_StopIter:
					currentBatchChrom = currentBatchChrom.cpu()
					currentBatchChrom_labels = currentBatchChrom_labels.cpu()
					numTotalChromSamples += len(currentBatchChrom_labels)
					currentBatchChromAndMito = currentBatchChrom.numpy()
					currentBatchChromAndMito_labels = currentBatchChrom_labels.numpy()
				
				elif not mito_StopIter and chrom_StopIter:
					currentBatchMito = currentBatchMito.cpu()
					currentBatchMito_labels = currentBatchMito_labels.cpu()
					numTotalMitoSamples += len(currentBatchMito_labels)
					currentBatchChromAndMito = currentBatchMito.numpy()
					currentBatchChromAndMito_labels = currentBatchMito_labels.numpy()
				else:
					currentBatchChrom = currentBatchChrom.cpu()
					currentBatchMito = currentBatchMito.cpu()
					currentBatchChrom_labels = currentBatchChrom_labels.cpu()
					currentBatchMito_labels = currentBatchMito_labels.cpu()

					numTotalMitoSamples += len(currentBatchMito_labels)
					numTotalChromSamples += len(currentBatchChrom_labels)
					currentBatchChromAndMito = np.concatenate((currentBatchChrom.numpy(), \
					currentBatchMito.numpy()))
					currentBatchChromAndMito_labels = np.concatenate((currentBatchChrom_labels.numpy() , \
					currentBatchMito_labels.numpy()))
				currentBatchChromAndMito_shuffled, currentBatchChromAndMito_labels_shuffled = shuffle(\
					currentBatchChromAndMito, currentBatchChromAndMito_labels)
				del currentBatchChromAndMito, currentBatchChromAndMito_labels
				currentBatchChromAndMito_shuffled_Variable = Variable(torch.FloatTensor(\
					currentBatchChromAndMito_shuffled))
				currentBatchChromAndMito_labels_shuffled_Variable =  Variable(\
					torch.LongTensor(currentBatchChromAndMito_labels_shuffled), requires_grad=False)

				if use_gpu:
					currentBatchChromAndMito_shuffled_Variable = currentBatchChromAndMito_shuffled_Variable.cuda()
					currentBatchChromAndMito_labels_shuffled_Variable = currentBatchChromAndMito_labels_shuffled_Variable.cuda()
				stage3 = time.time()


				################### Traning
				################### Traning
				################### Traning
				model.train(True)
				model.zero_grad()
				train_loss, train_accuracy, scores = compute_loss_accuracy(\
					data= currentBatchChromAndMito_shuffled_Variable,\
					label=currentBatchChromAndMito_labels_shuffled_Variable,\
					model = model, validation = True)
				if currentBatchNumber > 100:
					scheduler.step(sum(lossArray[-50:])/50, currentBatchNumber)

				if currentBatchNumber > 99:
					if sum(lossArray[-20:])/20 < minLoss:
						minLoss = sum(lossArray[-20:])/20
						minLossModel = model

				## logging and printing accuracy and loss
				stage4  = time.time()
				lossArray.append(train_loss.data.item())
				accArray.append(train_accuracy.data.item())

				valid_categoryAccCounter = [0 for k in range(2)]
				valid_categorySampleCounter = [0 for k in range(2)]
				for position, k in enumerate(currentBatchChromAndMito_labels_shuffled_Variable):
					valid_categorySampleCounter[k.data.item()] += 1
					if (k.data.item() == torch.max(scores, 1)[1].data[position]):
						if k.data.item() == 1:
							numCorrectMitoSamples += 1
						if k.data.item() == 0:
							numCorrectChromSamples += 1
						valid_categoryAccCounter[k.data.item()] += 1
				percentageCategory = (np.divide(valid_categoryAccCounter, valid_categorySampleCounter))
				percentageCategory[np.isnan(percentageCategory)] = 0


				train_loss.backward()
				optimizer.step()


				stage5 = time.time()
				if currentBatchNumber % 20 == 0:
					for param_group in optimizer.param_groups:
						current_lr = float(param_group['lr'])
					print("epoch: " , str(currentEpochMito), " | batch number: ", currentBatchNumber, \
					 " | start/current LR:", str(optim_lr),",", str(current_lr))
					 # " | reads left: ", len(train_chrom_dataset)," out of ", len(wholeData_Chrom))
					print("loss is: ", "{0:.4f}".format(\
						train_loss.data.item()) ,\
					 " \nand acc is: ", "{0:.4f}".format(\
						train_accuracy.data.item() ))
					print("acc for classes: ",percentageCategory) 
					unique, counts = np.unique(torch.max(scores, 1)[1].data.cpu().numpy(), return_counts=True)
					print("", dict(zip(unique, counts)))

				## running validation
				if currentBatchNumber % 100 == 0:
					del train_loss, train_accuracy, scores,\
						currentBatchChromAndMito_shuffled_Variable, currentBatchChromAndMito_labels_shuffled_Variable,\
						currentBatchChromAndMito_shuffled, currentBatchChromAndMito_labels_shuffled
					for test_iteration in range(batchesForTest):			
						try:
							currentBatchChrom, currentBatchChrom_labels = test_chrom_dataloader.__next__()
						except StopIteration:
							test_chrom_dataset = NanoporeDataset(torch.from_numpy(wholeData_Chrom_TEST),
													torch.from_numpy(wholeData_Chrom_labels_TEST),
													shuffle = shuffleDatasets, transform = transform)

							test_chrom_dataloader = iter(DataLoader(dataset=test_chrom_dataset,
																	   batch_size=batch_size, 
																	   shuffle=shuffleDatasets,
																	   drop_last = False,
																	   num_workers = num_of_workers, pin_memory = use_pin_memmory))
							currentBatchChrom, currentBatchChrom_labels = test_chrom_dataloader.__next__()
					
						try:
							currentBatchMito, currentBatchMito_labels = test_mito_dataloader.__next__()
						except StopIteration:
							test_mito_dataset = NanoporeDataset(torch.from_numpy(wholeData_Mito_TEST),
													torch.from_numpy(wholeData_Mito_labels_TEST),
													shuffle = shuffleDatasets, transform = transform)

							test_mito_dataloader = iter(DataLoader(dataset=test_mito_dataset,
																	   batch_size=batch_size, 
																	   shuffle=shuffleDatasets,
																	   drop_last = False,
																	   num_workers = num_of_workers, pin_memory = use_pin_memmory))
							currentBatchMito, currentBatchMito_labels = test_mito_dataloader.__next__()
						currentBatchChrom = currentBatchChrom.cpu()
						currentBatchMito = currentBatchMito.cpu()
						currentBatchChrom_labels = currentBatchChrom_labels.cpu()
						currentBatchMito_labels = currentBatchMito_labels.cpu()
						currentBatchChromAndMito = np.concatenate((currentBatchChrom.numpy(), \
						currentBatchMito.numpy()))
						currentBatchChromAndMito_labels = np.concatenate((currentBatchChrom_labels.numpy() , \
						currentBatchMito_labels.numpy()))
						
						currentBatchChromAndMito_shuffled, currentBatchChromAndMito_labels_shuffled = shuffle(\
							currentBatchChromAndMito, currentBatchChromAndMito_labels)
						del currentBatchChromAndMito, currentBatchChromAndMito_labels
						currentBatchChromAndMito_shuffled_Variable = Variable(torch.FloatTensor(\
							currentBatchChromAndMito_shuffled))
						currentBatchChromAndMito_labels_shuffled_Variable =  Variable(\
							torch.LongTensor(currentBatchChromAndMito_labels_shuffled), requires_grad=False)

						if use_gpu:
							currentBatchChromAndMito_shuffled_Variable = currentBatchChromAndMito_shuffled_Variable.cuda()
							currentBatchChromAndMito_labels_shuffled_Variable = currentBatchChromAndMito_labels_shuffled_Variable.cuda()

						################### Testing
						model.train(False)
						valid_loss, valid_accuracy, valid_scores = compute_loss_accuracy(\
							data= currentBatchChromAndMito_shuffled_Variable,\
							label=currentBatchChromAndMito_labels_shuffled_Variable,\
							model = model, validation = True)

						lossArrayTest.append(valid_loss.data.item())
						accArrayTest.append(valid_accuracy.data.item())

						valid_categoryAccCounter = [0 for k in range(2)]
						valid_categorySampleCounter = [0 for k in range(2)]
						for position, k in enumerate(currentBatchChromAndMito_labels_shuffled_Variable):
							valid_categorySampleCounter[k.data.item()] += 1
							if (k.data.item() == torch.max(valid_scores, 1)[1].data[position]):
								valid_categoryAccCounter[k.data.item()] += 1
						percentageCategory = (np.divide(valid_categoryAccCounter, valid_categorySampleCounter))
						percentageCategory[np.isnan(percentageCategory)] = 0
						allChromAccTest.append(np.multiply(percentageCategory, 100).astype(int))


					accArryForTest = accArryForTest+accArray[-batchesForTest:]
					lossArryForTest = lossArryForTest+lossArray[-batchesForTest:]



					print("VALIDATION START ===================")
					print("epoch: " , str(currentEpochMito), " | batch number: ", currentBatchNumber, \
					 " | reads left: ", len(train_chrom_dataset)," out of ", len(wholeData_Chrom))
					print("loss is: ", "{0:.4f}".format(\
						sum(lossArrayTest[-batchesForTest:])/batchesForTest) ,\
					 " \nand acc is: ", "{0:.4f}".format(\
						sum(accArrayTest[-batchesForTest:])/batchesForTest ))
					unique, counts = np.unique(torch.max(valid_scores, 1)[1].data.cpu().numpy(), return_counts=True)
					print("", dict(zip(unique, counts)))

					if currentBatchNumber > 150:
						os.remove(newFinetuneFolder +"/"+'trainLoss'+filenameEndingString+'.pdf')
						os.remove(newFinetuneFolder +"/"+'trainAcc'+filenameEndingString+'.pdf')
						os.remove(newFinetuneFolder +"/"+'testChromAcc'+filenameEndingString+'.pdf')
						os.remove(newFinetuneFolder +"/"+'testLoss'+filenameEndingString+'.pdf')
						os.remove(newFinetuneFolder +"/"+'testAcc'+filenameEndingString+'.pdf')
						

					filenameEndingString = "_Plot_batchSize"+str(batch_size)+"_epoch"+str(currentEpochMito)
					plt.plot(lossArray)
					lossPlt.savefig(newFinetuneFolder +"/"+'trainLoss'+filenameEndingString+'.pdf')
					plt.clf()
					# lossPlt = plt.figure()
					plt.plot(accArray)
					lossPlt.savefig(newFinetuneFolder +"/"+'trainAcc'+filenameEndingString+'.pdf')
					plt.clf()
					# lossPlt = plt.figure()
					plt.plot(allChromAccTest)
					lossPlt.savefig(newFinetuneFolder +"/"+'testChromAcc'+filenameEndingString+'.pdf')
					plt.clf()
					# lossPlt = plt.figure()
					plt.plot(lossArryForTest)
					plt.plot(lossArrayTest)
					lossPlt.savefig(newFinetuneFolder +"/"+'testLoss'+filenameEndingString+'.pdf')
					plt.clf()
					plt.plot(accArryForTest)
					plt.plot(accArrayTest)
					lossPlt.savefig(newFinetuneFolder +"/"+'testAcc'+filenameEndingString+'.pdf')
					plt.clf()


					if currentBatchNumber > 100 and currentBatchNumber % 100 == 0:
							save_path = '{}/{}'.format(newFinetuneFolder, "Nanopore_model.pth")
							save_path_notMin = '{}/{}'.format(newFinetuneFolder, "Nanopore_model_notMin.pth")
							torch.save(minLossModel, save_path)
					print("VALIDATION END ===================")
					del valid_loss, valid_accuracy, valid_scores,\
						currentBatchChromAndMito_shuffled_Variable, currentBatchChromAndMito_labels_shuffled_Variable,\
						currentBatchChromAndMito_shuffled, currentBatchChromAndMito_labels_shuffled
					model.train(True)
				 ## end of training and validation

				stage7 = time.time()
				Stage1Time += stage1 - startTime
				Stage2Time += stage2 - stage1
				Stage3Time += stage3 - stage2
				Stage4Time += stage4 - stage3
				Stage5Time += stage5 - stage4
				Stage6Time += stage7 - stage5
				totalTime += stage7 - startTime

				##uncomment to see timing during different stages of training

				# print("first: ", Stage1Time)
				# print("second: ", Stage2Time)
				# print("third: ", Stage3Time)
				# print("fourth: ", Stage4Time)
				# print("fifth: ", Stage5Time)
				# print("sixsth: ", Stage6Time)
				# print("TOTAL: ", totalTime)




			print(numTotalMitoSamples ,
			numTotalChromSamples ,
			numCorrectMitoSamples ,
			numCorrectChromSamples )

			summaryFile.write(str(currentTestingModelFolder) + ", " +str(numTotalMitoSamples) + ", " +str(numCorrectMitoSamples) + ", " +str(numTotalChromSamples) + ", "+ str(numCorrectChromSamples)+',\n')


if __name__ == '__main__':
	parser = argparse.ArgumentParser('Train the model.')
	parser.add_argument('--hidden-size', required=True, type=int,
						help='The number of hidden units')
	parser.add_argument('--batch-size', required=True, type=int,
						help='The size of each batch')
	parser.add_argument('--max-iter', required=True, type=int,
						help='The maximum iteration count')
	parser.add_argument('--gpu', default=False, action='store_true',
						help='The value specifying whether to use GPU')
	args = parser.parse_args()
	main()
