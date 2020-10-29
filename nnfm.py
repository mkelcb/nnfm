# -*- coding: utf-8 -*-
"""

@author: mk23
"""

# DLpred manager
# where /how to store reverse complements?
# dont want to worry about it later, so once loaded in they should just behave like regular training labels/data
# when loading / handling the data:
# should be easy, it should NOT be interlaced: (IE Label1,Label1_rev,Label2,Label2_rev) but rather (label1,label2,lave1_rev,lavel2_rev


# Shystem
import argparse
import numpy as np
import time
import os

import random

import pandas as pd
import matplotlib
import platform
if platform.system().find('Windows') == -1 :
    print("Matlab uses Agg as we are on a *nix")
    matplotlib.use('Agg')

#import matplotlib.pyplot as plt   
from pathlib import Path
# Hyperopt


# Pytorch
import torch
import torch.nn as nn


# DLPred vars
from nnfm_common import   loadLabels, loadChroms, printElapsedTime, build_model, model_diagnostics, SeqBinDataset, danQ, TBiNet, calculateDataStats, calculateDataStats_perChannel, writeMatrixToDisk, loadMatrixFromDisk, writeYMeanSds, zscore, y_standardise, maxAbs, reverse_maxAbs, loadYMaxAbs, writeYMaxAbs, y_standardise_maxAbs, loadAllMotifData, loadTargetNames, getAccuracy, writeAccuracy, evaluateModel,DEEPSea, modelToDevince, writeSortedYs

from nnfm_trainer import learn, getModel



##################################################################

###############################################################################
# Global vars
###############################################################################
device = None

y= None
y_valid= None
pheno_is_binary = False


#args  = None
###############################################################################

def process(args) :
    global y; global y_valid; global pheno_is_binary 
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed) # set this here, as the order of the loaders needs to be 'fixed' too
    os.makedirs(os.path.dirname(args.out), exist_ok=True) # Make the output directory if it doesn't exist.

    ###########################################################################
    # (I) LOAD: load, preprocess and minibatch the data
    ##########
    print("(I) Data Load", flush = True)
    # 1) load entire chrom binaries for both training/validation: load all chromosomes in and store them into dictionaries  as:  training_chroms[chrom]
    training_chroms = loadChroms(args.trainSet, args.sequenceLoc)
    valid_chroms = loadChroms(args.validSet, args.sequenceLoc)

    # 2) load labels
    targets = None
    if args.targetNamesloc is not None: targets = loadTargetNames(args.targetNamesloc)
    y       ,training_labels_dict = loadLabels(args.labels, training_chroms, shuffleLabels = args.permutePhenos) # args.reverseCompliment, disable this for loading, to save on RAM
    y_valid , valid_labels_dict   = loadLabels(args.labels, valid_chroms)
    args.n_genomic_features = y.shape[1] # the number of genomic features is the length of y, IE this is the 919 (or 690) TFs

    #3) standardise data: 
    if len( np.unique(y) ) <= 2 :  #check if the labels are binary, as if yes, then we dont want to standardise it
        print("pheno is inferred to be binary")
        pheno_is_binary = True
    else  :
        print("pheno is inferred to be continuous, standardising")
        pheno_is_binary = False
        y, y_maxabs = maxAbs(y)
        y_valid = y_standardise_maxAbs(y_valid,y_maxabs)  # apply same standardisation onto the training labels
        writeYMaxAbs(args.out, y_maxabs ) # write the maxabs value to disk so that it could be loaded for test sets

    # print("ALL PHENO IS TEMPORARILY TREATED AS CONTINUOUS")
    # pheno_is_binary = False
    # y, y_maxabs = maxAbs(y)
    # y_valid = y_standardise_maxAbs(y_valid,y_maxabs)  # apply same standardisation onto the training labels
    # writeYMaxAbs(args.out, y_maxabs ) # write the maxabs value to disk so that it could be loaded for test sets


    # 4a) load Motifs
    if args.motifLoc is not None : 
        training_motifs = loadAllMotifData(training_chroms, args.motifLoc)
        valid_motifs = loadAllMotifData(valid_chroms, args.motifLoc)
    else : training_motifs = valid_motifs = None
    
    # 4b) load Weights
    if args.weights is not None : 
        print("loading weights")
        training_weights, _ = loadLabels(args.weights, training_chroms, extension = ".weight")
        valid_weights, _ = loadLabels(args.weights, valid_chroms, extension = ".weight")
    else : training_weights = valid_weights = None  

    # 5) determine device: do this BEFORE setting up the data, as the dataLoaders will want to 'pin memory' for the GPU, which default to GPU0 if we dont specify otherwise
    global device
    device = torch.device("cuda:"+str(args.device) if torch.cuda.is_available() and args.gpu > 0 else "cpu")

    args.accumulation_steps = max(1,args.gradient_batch_size // args.batch_size ) # number of accumulation steps is based on the effective batch size we want for the gradient calculation ( do this before upscalign batch size)
    print("gradient batch_size:", args.gradient_batch_size, " / batch_size:", args.batch_size, "/ accumulation_steps worked out to be", args.accumulation_steps)
    workers= max(1,args.gpu *4) # number of workers is 4x the num GPUs but at least 1
    
    if platform.system()=='Windows': workers=0 # Windows 10 has bug preventing multiprocessing creating Broken Pipe: https://github.com/pytorch/pytorch/issues/2341   and  https://discourse.pymc.io/t/multiprocessing-windows-10-brokenpipeerror-errno-32-broken-pipe/2259/9
    if torch.cuda.is_available() and args.gpu > 0 and args.altModel != danQ and args.altModel != TBiNet: # danQ/TBiNet are RNNs which will NOT use multiGPUs, so scaling up the batch size will just cause VRAM OOM 
        args.batch_size = args.batch_size *args.gpu # scale up the batch size for the number of GPUs we will use


    # 6) Setup Data   
    trainingData = SeqBinDataset(y      , training_labels_dict, training_chroms, args.seqLength, augment_shift = args.augment_shift, augment_rc = args.reverseCompliment, castToTensor = True, half = args.half,motifChroms = training_motifs, motifRange = args.motifRange, protectedRange = args.protectedRange, weights = training_weights, insertA1 = args.insertA1)
 
    # dont standardise the genotype data for now, as we are using batchnorms
    if args.standardiseInput :
        print("calculating training data means"); data_means, data_sds = loadOrCalcDataStats(args, trainingData)  #data_means, data_sds = calculateDataStats(trainingData) # calculateDataStats # calculateDataStats_perChannel# obtain the trainin
        trainingData.data_means = data_means ; trainingData.data_sds = data_sds
        writeMatrixToDisk(args.out+"_data_means",data_means);writeMatrixToDisk(args.out+"_data_sds",data_sds);
    validData =    SeqBinDataset(y_valid, valid_labels_dict   , valid_chroms   , args.seqLength, augment_shift = False, augment_rc = False, castToTensor = True, half = args.half,  data_means = trainingData.data_means, data_sds=  trainingData.data_sds,motifChroms = valid_motifs, motifRange = args.motifRange, protectedRange = args.protectedRange, weights = valid_weights, insertA1 = args.insertA1)

    # create loaders
    trainLoader = torch.utils.data.DataLoader(trainingData, batch_size = args.batch_size, shuffle = True,pin_memory = args.gpu > 0,num_workers = workers,drop_last  = True)  # drop_last True, as if the last batch has a size of 1, then batchnorm would die
    validLoader = torch.utils.data.DataLoader(validData, batch_size = args.batch_size, shuffle = True,pin_memory = args.gpu > 0,num_workers = workers,drop_last  = False) # no need to drop last for validation, as bnorm is off there anyway
    
    train_chanceLevel_loss = meanLevelLoss(y,len(trainLoader)) ; valid_chanceLevel_loss = meanLevelLoss(y_valid,len(validLoader))

   
    ###########################################################################
    # II) Model Build: find or load the best hyper param settings and build a model
    ##########
    #print("(II) Model Selection", flush = True)


    ###########################################################################
    # III) Train model:
    ###################
    print("(III) Train Model on", device, flush = True)

    # 1) train non-linear model
    torch.cuda.empty_cache()

    if modelAlreadyRun(args) == False :
        if args.transfer is not None: model = build_transferLearningModel(args, device)
        else : model = build_model(args, device)
        model_diagnostics(getModel(model),args.out)
        line="Trainset len/batches: " +str(len(y) ) + "/"+ str(len(trainLoader) )  +" / Validset len/batches: " +str(len(y_valid) ) + "/" + str(len(validLoader) ) + " | chance level prediction training Loss: " + str(round( train_chanceLevel_loss,3))   + " | validation loss: " + str(round(valid_chanceLevel_loss ,3) )   ; print( line)
        with open(args.out+"_chance", "w") as file: file.write(line  + "\n")
        trainModel(args, model,device, trainLoader, validLoader, targets)
        del model  # relese VRAM
        torch.cuda.empty_cache()


def build_transferLearningModel(args, device):
    print("Transfer learning")
    # 1) build original DS model
    altModel_orig = args.altModel
    n_genomic_features_orig = args.n_genomic_features
    args.n_genomic_features = 690
    args.altModel = DEEPSea
    model = build_model(args, device, mapModelToDevice = False) # do NOT parallelise it yet or move it to the GPU
    
    # 2) load Deepsea weights
    model.load_state_dict(torch.load(args.transfer,  map_location='cpu')) # load it to the CPU first
    
    # 3) Freeze weights
    for param in model.parameters(): param.requires_grad = False
    
    # 4) replace original 'head' of the model with a new untrained NNP head
    args.altModel = altModel_orig  # reset original model choice
    args.n_genomic_features = n_genomic_features_orig
    NNPhead = build_model(args, device, mapModelToDevice = False)
    model.classifier = NNPhead.classifier
 
    
    # 5) once we have the final model move it to the correct device and parallelise it
    model = modelToDevince(model,args,device)
   
    return(model)
 

    
    
def trainModel(args, model,device, trainLoader, validLoader, targets = None, postFix = "") :
    # 2) train model for the first time 
    start = time.time()
    saveModelLocation = None
    if args.saveWeights is not None : saveModelLocation = args.saveWeights + postFix
    results = learn(model,device, trainLoader, validLoader, saveModelLocation = saveModelLocation, epochMaxImproveThreshold = args.epochMaxImproveThreshold, learnRate = args.learnRate, half = args.half,accumulation_steps = args.accumulation_steps, l2= args.l2, decayRate = args.LRdecay, sgd = args.sgd, pheno_is_binary = pheno_is_binary)  # , debugOut =args.out +"debug" + postFix, debugCallback = debugCallback
    end = time.time(); printElapsedTime(start,end, "training model took: ")
    results_its = results["results"]  

    # 3)   write out the best epoch (this is 0 based)
    with open( args.out + postFix + "best_epoch.txt", "w") as file: file.write("best_epoch=" + str(results['results']['lowestLoss_epoch']) ) # write out the early stop epoch used

    # 4) Save model params
    if args.saveWeights is not None : 
        lowestLoss_epoch =results['results']['lowestLoss_epoch']
        for i in range( len(results['results']["epochs"]) ) :
            epoch = results['results']["epochs"][i]
            
            if epoch == lowestLoss_epoch : # rename the best model 
                print("highest performing model at epoch",lowestLoss_epoch, " saved to:",args.saveWeights+ postFix)
                os.rename(args.saveWeights+ postFix + str(lowestLoss_epoch), args.saveWeights+ postFix)
            elif epoch > 0 : # delete the rest, except model0, the one with no training as a reference
                os.remove(args.saveWeights+ postFix + str(epoch))


    ###########################################################################
    # IV) Output diagnostics:
    #########################
    print("(IV) Output", flush = True)

    # 1) write out summary results of the training iterations
    fileName = args.out + postFix + "nn_results.txt"
    with open(fileName, "w") as file:      
        line = "epochs"
        if "train_loss" in results_its: line = line + "\t" + "train_loss"
        if "valid_loss" in results_its: line = line + "\t" + "valid_loss"

        file.write(line  + "\n")
         
        for i in range( len(results_its["epochs"])  ):
            line = str(results_its["epochs"][i]) 
            if "train_loss" in results_its: line = line + "\t" + str(results_its["train_loss"][i])
            if "valid_loss" in results_its: line = line + "\t" + str(results_its["valid_loss"][i])

            file.write(line + "\n")            
        
    # 2) generate learning curve plots of the results
    if len(results_its["epochs"]) > 0 :
        plotNNtraining(results_its, args.out + "trainplot_loss"+postFix, training_metric="train_loss", valid_metric="valid_loss", acc_measure = "BCE")

    # 3a) output validation accuracy: of the FINAL model
    outputAccuracy(args, model,device, validLoader, targets, postFix = "_final")
    
    # 3b) output the validation accuracy for the BEST model
    modelWeightsToload = args.saveWeights+ postFix
    model = build_model(args, device, mapModelToDevice = False) # do NOT parallelise it yet or move it to the GPU
    model.load_state_dict(torch.load(modelWeightsToload,  map_location='cpu')) # load it to the CPU first
    model = modelToDevince(model,args,device)
 
    
    
    # if torch.cuda.is_available() and args.gpu > 0 : 
        
    #     if args.device != -1 : # if a specific device Id was requested, we try to map weights there
    #         model.load_state_dict(torch.load(modelWeightsToload,  map_location="cuda:"+str(args.device)))
    #     else :model.load_state_dict(torch.load(modelWeightsToload)) # otherwise attempt reconstruct the model to same devices as they were trained on, IE for multi GPUs
        
    # else :  model.load_state_dict(torch.load(modelWeightsToload,  map_location='cpu'))
    outputAccuracy(args, model,device, validLoader, targets, postFix = "_best")
    
    
def outputAccuracy(args, model,device, validLoader, targets, postFix = "") :
    
    orig_augment_rc = validLoader.dataset.augment_rc
    orig_force_rc = validLoader.dataset.force_rc
    # predict validation accuracy 

    yhat, y, weights, indices = evaluateModel(args, model,device, validLoader)
    overall_max_R, overall_max_S, overall_max_r, overall_mean_R, overall_mean_S, overall_mean_S_p, weighted_R_mean, weighted_S_mean, y_Rs, y_Ss, y_S_ps, y_rs, y_r_ps, overall_mean_r, overall_mean_r_p, weighted_r_mean, sumPP = getAccuracy(y,yhat, weights) 
    writeAccuracy(args.out + postFix,overall_max_R, overall_max_S, overall_max_r,overall_mean_R, overall_mean_S, overall_mean_S_p, weighted_R_mean, weighted_S_mean, y_Rs, y_Ss, y_S_ps, y_rs, y_r_ps, overall_mean_r, overall_mean_r_p, weighted_r_mean , sumPP,targets = targets)

    # write it out again, just to see how much the stochastic sampling affects accuracy
    yhat, y, weights, indices = evaluateModel(args, model,device, validLoader)
    overall_max_R, overall_max_S, overall_max_r, overall_mean_R, overall_mean_S, overall_mean_S_p, weighted_R_mean, weighted_S_mean, y_Rs, y_Ss, y_S_ps, y_rs, y_r_ps, overall_mean_r, overall_mean_r_p, weighted_r_mean, sumPP = getAccuracy(y,yhat, weights) 
    writeAccuracy(args.out+"_again" + postFix,overall_max_R, overall_max_S, overall_max_r,overall_mean_R, overall_mean_S, overall_mean_S_p, weighted_R_mean, weighted_S_mean, y_Rs, y_Ss, y_S_ps, y_rs, y_r_ps, overall_mean_r, overall_mean_r_p, weighted_r_mean, sumPP ,targets = targets)

    # now do the F, RC and average
    validLoader.dataset.augment_rc = False
    validLoader.dataset.force_rc = False
    yhats_all, b_labels_all, weights, indices  = evaluateModel(args, model,device, validLoader)
    writeSortedYs(args.out+"_yhat" + postFix, yhats_all, indices)
    writeSortedYs(args.out+"_y" + postFix, b_labels_all, indices)
    
    validLoader.dataset.force_rc  = True
    yhats_all_RC, b_labels_all_RC, weights_RC, indices_RC  = evaluateModel(args, model,device, validLoader)
    writeSortedYs(args.out+"_y_RC" + postFix, b_labels_all_RC, indices_RC)
    writeSortedYs(args.out+"_yhat_RC" + postFix, yhats_all_RC, indices_RC)
    
    overall_max_R_F, overall_max_S_F, overall_max_r_F, overall_mean_R_F, overall_mean_S_F, overall_mean_S_p_F, weighted_R_mean_F, weighted_S_mean_F, y_Rs_F, y_Ss_F, y_S_ps_F, y_rs_F, y_r_ps_F, overall_mean_r_F, overall_mean_r_p_F, weighted_r_mean_F, sumPP = getAccuracy(b_labels_all,yhats_all, weights) 
    overall_max_R_RC, overall_max_S_RC, overall_max_r_RC, overall_mean_R_RC, overall_mean_S_RC, overall_mean_S_p_RC, weighted_R_mean_RC, weighted_S_mean_RC, y_Rs_RC, y_Ss_RC, y_S_ps_RC, y_rs_RC, y_r_ps_RC, overall_mean_r_RC, overall_mean_r_p_RC, weighted_r_mean_RC, sumPP = getAccuracy(b_labels_all_RC,yhats_all_RC, weights_RC)
    writeAccuracy(args.out+"_F" + postFix,overall_max_R_F, overall_max_S_F, overall_max_r_F, overall_mean_R_F, overall_mean_S_F, overall_mean_S_p_F, weighted_R_mean_F, weighted_S_mean_F, y_Rs_F, y_Ss_F, y_S_ps_F, y_rs_F, y_r_ps_F, overall_mean_r_F, overall_mean_r_p_F, weighted_r_mean_F , sumPP,targets = targets)
    writeAccuracy(args.out+"_RC" + postFix,overall_max_R_RC, overall_max_S_RC, overall_max_r_RC, overall_mean_R_RC, overall_mean_S_RC, overall_mean_S_p_RC, weighted_R_mean_RC, weighted_S_mean_RC, y_Rs_RC, y_Ss_RC, y_S_ps_RC, y_rs_RC, y_r_ps_RC, overall_mean_r_RC, overall_mean_r_p_RC, weighted_r_mean_RC, sumPP ,targets = targets)

    # average
    overall_mean_R =(overall_mean_R_F + overall_mean_R_RC )/2.
    overall_mean_S =(overall_mean_S_F + overall_mean_S_RC)/2.
    overall_mean_S_p =(overall_mean_S_p_F+ overall_mean_S_p_RC) /2.
    weighted_R_mean =(weighted_R_mean_F + weighted_R_mean_RC) /2.
    weighted_S_mean =(weighted_S_mean_F + weighted_S_mean_RC) /2.
    y_Rs   =(( np.array(y_Rs_F  ) + np.array(y_Rs_RC)   )/2. ).tolist()
    y_Ss   =(( np.array(y_Ss_F  ) + np.array(y_Ss_RC)   )/2. ).tolist()
    y_S_ps =(( np.array(y_S_ps_F) + np.array(y_S_ps_RC) )/2. ).tolist()
    overall_mean_r  =(overall_mean_r_F+ overall_mean_r_RC) /2.
    overall_mean_r_p  =(overall_mean_r_p_F+ overall_mean_r_p_RC) /2.
    weighted_r_mean =(weighted_r_mean_F+ weighted_r_mean_RC) /2.
    y_rs   =(( np.array(y_rs_F  ) + np.array(y_rs_RC)   )/2. ).tolist()
    y_r_ps =(( np.array(y_r_ps_F) + np.array(y_r_ps_RC) )/2. ).tolist()
    overall_max_R =(overall_max_R_F + overall_max_R_RC )/2.
    overall_max_S =(overall_max_S_F +overall_max_S_RC )/2.
    overall_max_r =(overall_max_r_F + overall_max_r_RC )/2.
    
    writeAccuracy(args.out + postFix,overall_max_R, overall_max_S, overall_max_r,overall_mean_R, overall_mean_S, overall_mean_S_p, weighted_R_mean, weighted_S_mean, y_Rs, y_Ss, y_S_ps, y_rs, y_r_ps, overall_mean_r, overall_mean_r_p, weighted_r_mean, sumPP ,targets = targets)
    
    # reset data to be how it was before
    validLoader.dataset.augment_rc = orig_augment_rc
    validLoader.dataset.force_rc = orig_force_rc
    



        
 
        


###############################################################################
# Plotting
###############################################################################  


#data = results
# location = "../../../0cluster/results/local/test"
#acc_measure = "r^2"

def plotNNtraining(data,location, training_metric, valid_metric, acc_measure = "MSE", epoch_offset = 0) : # as the epoch's first training loss is much higher as that reflects before training accuracy, therefore we want to start at second epoch

    content = None
    cols = list()
    
    if platform.system().find('Windows') == -1 :
        print("Matlab uses Agg as we are on a *nix")
        matplotlib.use('Agg')
        
    import matplotlib.pyplot as plt
    
    if training_metric in data:
        #print("traing exists")
        cols.append('Traning')
        if(content is None) : content = data[training_metric][epoch_offset:len(data[training_metric])]
        else : content = np.column_stack( (content, data[training_metric][epoch_offset:len(data[training_metric])] ) )
        
    if valid_metric in data:
        #print("test exists")
        cols.append('Validation')
        if(content is None) : content = data[valid_metric][epoch_offset:len(data[valid_metric])]
        else : content = np.column_stack( (content, data[valid_metric][epoch_offset:len(data[valid_metric])]  ) ) 
        
        
    df = pd.DataFrame(content, index=data["epochs"][epoch_offset:len(data["epochs"])], columns=cols )
    

    #df = df.cumsum()
    
    plt.figure()
    
    ax = df.plot(title = "NN learning curve") 
    ax.set_xlabel("epochs")
    ax.set_ylabel(acc_measure)
    

    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(location + '.eps', format='eps', dpi=1000)
    fig.savefig(location + '.png', dpi=300)


###############################################################################
# Utils
###############################################################################  

def loadOrCalcDataStats(args, trainingData): # util to load pre-calculated data means, as this may be an expensive operation
    pastDataStats = Path(args.out+"_data_means.bin")
    if pastDataStats.is_file():
        print("loading existing data stats")
        data_means = loadMatrixFromDisk(args.out+"_data_means"); data_sds = loadMatrixFromDisk(args.out+"_data_sds");
    else : 
        print("no data stats yet, calculating new")
        data_means, data_sds = calculateDataStats(trainingData)
    
    return(data_means, data_sds)


def modelAlreadyRun (args, postFix = "") : # util, to determine if a model has already run,
    if args.saveWeights is not None :    
        pastModel = Path(args.saveWeights + postFix)
        if pastModel.is_file():
            print(postFix,"model already run")
            return(True)
        else : return(False)
    else : return(False)
 
    
def meanLevelLoss(y, num_batches =1) : #calculates the loss for a mean level prediction
    mean_y = np.mean(y, axis=0, keepdims=True)
    meanPrediction = np.ones(y.shape, dtype=np.float32) * mean_y

   
    meanPrediction = torch.Tensor( meanPrediction)
    if pheno_is_binary : criterion = nn.BCELoss()
    else :  criterion = nn.MSELoss()
    y_torch =  torch.Tensor(y)

    
    # the loss is averaged along both axes for torch, so if we want the un-averaged version, we multiply by their product
    loss = criterion(meanPrediction,y_torch ).item()  * ( y.shape[0] *  y.shape[1] )  /num_batches  # calculate error
    return(loss)
    #loss_torch = criterion(meanPrediction,y_torch ).item() # 0.0012115546269342303
    #manualLoss = np.sum( (y -meanPrediction)**2 ) / ( y.shape[0] *  y.shape[1] ) # 0.001211554640314582


    
    
    # when/do we standardise the the 4channel One hots??? Check DanQ / Selene how they do it

    # DanQ apparently only standardised their 'y' not their X???
    # as I only have one y, it is all on the same scale, I dont need to standardise them

    # same for DEEPSEA:
    # https://www.nature.com/articles/nmeth.3547#Sec11
    # "All features were standardized to mean 0 and variance 1 before training. Unequal positive and negative training sample sizes were balanced with sample weights."



###############################################################################
# Hyperopt END
###############################################################################  


##################################################


# out="C:/softwares/Cluster/GIANT/miniPRS/dlpred/nnp/results/0/"
# labels='C:/softwares/Cluster/GIANT/miniPRS/dlpred/nnfm/labels/'
# sequenceLoc='C:/softwares/Cluster/GIANT/miniPRS/dlpred/deepsea/chroms/'
# trainSet='C:/softwares/Cluster/GIANT/miniPRS/dlpred/deepsea/folds/train_0'
# validSet='C:/softwares/Cluster/GIANT/miniPRS/dlpred/deepsea/folds/valid_0'
# motifLoc='C:/softwares/Cluster/GIANT/miniPRS/dlpred/nnp/motifs/'
# seed=42
# seqLength='1000'
# reverseCompliment=False

# args = parser.parse_args(['--out', out,  '--labels',labels, '--motifLoc', motifLoc, '--sequenceLoc', sequenceLoc, '--trainSet', trainSet, '--validSet', validSet, '--seqLength', seqLength])
# args.epochMaxImproveThreshold=1
# args.targetNamesloc = 'C:/softwares/Cluster/GIANT/miniPRS/dlpred/nnp/pheno'  
# args.saveWeights='C:/softwares/Cluster/GIANT/miniPRS/dlpred/deepsea/results/model'

# args.weights = 'C:/softwares/Cluster/GIANT/miniPRS/dlpred/nnp/weights/'

def runMain(args) :
    args.earlystop = args.earlystop == 1
    args.bnorm = args.bnorm == 1
    args.half = args.half == 1
    args.standardiseInput = args.standardiseInput == 1
    args.sgd = args.sgd == 1

    args.reverseCompliment = args.reverseCompliment == 1
    args.augment_shift = args.augment_shift == 1
    args.permutePhenos = args.permutePhenos == 1
    args.insertA1 = args.insertA1 == 1

    
    

    print('Fitting NNP on training set',  args.trainSet, "learnRate:" , args.learnRate, "/ sequences loaded from",  args.sequenceLoc, " / labels:",args.labels, "/ .permutePhenos:", args.permutePhenos, "/ augment_shift:", args.augment_shift,  "/ reverseCompliment:", args.reverseCompliment, "/ Pytorch:", torch.__version__)    
    process(args)
    

   
if __name__ == '__main__':   
    parser = argparse.ArgumentParser()

    parser.add_argument("--out",required=True, help='an output location is always required')
    parser.add_argument("--seqLength",required=False, help='the length of the sequences', default=1000, type=int)  
    parser.add_argument("--sequenceLoc",required=True, help='location of sequence chrom, in files in .sbin and .sid format ')  
    parser.add_argument("--trainSet",required=True, help='list of chroms for training set')  
    parser.add_argument("--validSet",required=True, help='list of chroms for validation set')  
    parser.add_argument("--labels",required=True, help='location of the per chrom labels in .label format')  
    parser.add_argument("--targetNamesloc",required=False, help='the location of the target class names')  

    parser.add_argument("--motifLoc",required=False, help='folder for the motifs')  
    parser.add_argument("--protectedRange",required=False, help='location around target for which motifs cannot be inserted, default 500', default=500, type=int)  
    parser.add_argument("--motifRange",required=False, help='The left/right range around the index to look for motifs, default 1000000', default=1000000, type=int)  
    
    parser.add_argument("--weights",required=False, help='folder for the weights to use weighted MSE')  

    
    parser.add_argument("--transfer",required=False, help='Transfer learning the location from where the other models weights are loaded from') 
    parser.add_argument("--transferLevel",required=False, help='The number of layers for which transfer learning is used for, default 1, (the lowest conv layer)', default=1, type=int) 
 

    parser.add_argument("--standardiseInput",required=False, help='if predictor input data should be standardised (1) or not (0), default 0', default=0, type=int)   

    parser.add_argument("--sgd",required=False, help='if SGD should be used (1) or ADAM (0), default 0', default=0, type=int)   


    parser.add_argument("--seed",required=False, help='random seed for dividing the chroms', default=42, type=int)  
    parser.add_argument("--batch_size",required=False, help='the size of the minibatches, default :64', default=64, type=int)        # 
    parser.add_argument("--gradient_batch_size",required=False, help='effective size of minibatches used for gradient calculation, default :64', default=64, type=int)        # 
    

    
    parser.add_argument("--gpu",required=False, help='the number of gpus to be used. 0 for cpu.', default=0, type=int)        # 
     # 

    parser.add_argument("--widthReductionRate", default=1, help='The rate at which the network "thins" IE if we start at 1000 neurons in layer 1, then at rate of 1 (default), we half it every layer, with a rate of 2, it will half every second layer Ie we will get two layers with 1000 units each, and then two 500 units etc', type=int) 

     # 

    parser.add_argument("--momentum",required=False, help='momentum used for the optimizer. default is 0.9', default=0.9, type=float)        # 
    parser.add_argument("--learnRate",required=False, help='learnRate used for the optimizer. default is 0.001', default=0.001, type=float)        # 
    parser.add_argument("--LRdecay",required=False, help='Learning rate decay, default 0.96 (to disable set it to -1)', default=-1, type=float)        # 

    parser.add_argument("--epochMaxImproveThreshold",required=False, help='Max number of epochs until no improvement before stopping. default is 12', default=12, type=int)        #  


    parser.add_argument("--earlystop",required=False, help='if early stop mechanism is to be applied (default True)', default=1, type=int)       
    parser.add_argument("--bnorm",required=False, help='if batchnorm (1, default) or group norm is to be used ', default=1, type=int)   
    parser.add_argument("--reverseCompliment",required=False, help='If reverse complement of the sequences are requested (default True)', default=0, type=int)  
    parser.add_argument("--augment_shift",required=False, help='if augmentation via shifting the sequence is enabled (default False)', default=0, type=int)   


    parser.add_argument("--device",required=False, help='the GPU device used to host the master copy of the model, default 0', default=0, type=int)   

    parser.add_argument("--permutePhenos",required=False, help='if phenotypes should be permuted (1) or not (0), default 0', default=0, type=int)   


    parser.add_argument("--saveWeights",required=False, help='the location of where the trained weights should be saved at, if not specified they wont be saved') # where we wnt to save weights
    parser.set_defaults(func=runMain)
    
    parser.add_argument("--half",required=False, help='if FP16 should be used (default no)', default=0, type=int)   


    parser.add_argument("--l2",required=False, help='l2 regularization', default=0.0, type=float)        # 
  
    parser.add_argument("--altModel",required=False, help='if the DanQ/Basset/DEEPSea, Basset Linear or NNP model is to be used, (for 1,2,3, 4,5 or 6) , default= 3  (= DEEPSea)', default=3, type=int)   


    parser.add_argument("--hidAct",required=False, help='the activation function, may be overriden by the best_pars', default=2, type=int)   

    parser.add_argument("--insertA1",required=False, help='if the effect-allele should be inserted (1, default) or not (1)', default=1, type=int)       
 

    args = parser.parse_args()
    args.func(args)



