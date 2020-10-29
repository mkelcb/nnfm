# -*- coding: utf-8 -*-
"""

@author: mk23
"""

# DSpred test
# this assumes that the test is preprocessed files with labels and sbins
# will write an alternative file that takes in text sequences with no labels
# and converts them into sbin and produces a prediction after optional padding

# System
import argparse
import numpy as np
import os
import random
import platform

from sklearn import metrics
import pandas as pd

# Pytorch
import torch



# DLPred vars
from nnfm_common import loadLabels, loadChroms, build_model,SeqBinDataset, danQ, TBiNet, loadMatrixFromDisk, loadTargetNames,getAccuracy, writeAccuracy, evaluateModel,loadYMaxAbs, loadAllMotifData, y_standardise_maxAbs, writeSortedYs, writeSortedPredictorData
#from nnpred_trainer import setModelMode


###############################################################################
# Global vars
###############################################################################
device = None
pheno_is_binary = False
###############################################################################

def process(args) :
    global pheno_is_binary ;
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed) # set this here, as the order of the loaders needs to be 'fixed' too

    os.makedirs(os.path.dirname(args.out), exist_ok=True) # Make the output directory if it doesn't exist.

    ###########################################################################
    # (I) LOAD: load, preprocess and minibatch the data
    ##########
    print("(I) Data Load", flush = True)
    # 1) load entire chrom binaries  for testset
    test_chroms = loadChroms(args.testSet, args.sequenceLoc)

    # 2) load labels
    targets = None
    if args.targetNamesloc is not None: targets = loadTargetNames(args.targetNamesloc)
    targets = loadTargetNames(args.targetNamesloc) # load the actuall classes (eg 'NHEK|H3K27me3|None' , 'NHEK|H3K27ac|None' etc)
    y_test, test_labels_dict = loadLabels(args.labels, test_chroms)
    args.n_genomic_features = y_test.shape[1] # the number of genomic features is the length of y, IE this is the 919 (or 690) TFs

    
    # 3) standardise data:
    # check if the mean of the labels is ~1, as if yes, then we dont want to standardise it
    y_maxabs = None
    if len( np.unique(y_test) ) <= 2 : 
        print("pheno is inferred to be binary")
        pheno_is_binary = True
    else  :
        print("pheno is inferred to be continuous, standardising")
        pheno_is_binary = False
        y_maxabs = loadYMaxAbs(args.loadWeights+ ".ymaxabs")
        y_test = y_standardise_maxAbs(y_test,y_maxabs)  # need to standardise this, as we apply the reverse transform to both y and yhat in evaluateModel
        
        
    # 4) load Motifs
    if args.motifLoc is not None :  test_motifs = loadAllMotifData(test_chroms, args.motifLoc)
    else : test_motifs = None
    
    # 4b) load Weights
    if args.weights is not None :   test_weights, _ = loadLabels(args.weights, test_chroms, extension = ".weight")
    else : test_weights = None  


    # 5) determine device: do this BEFORE setting up the data, as the dataLoaders will want to 'pin memory' for the GPU, which default to GPU0 if we dont specify otherwise
    global device
    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu > 0 else "cpu") # "cuda:"+str(args.device)

    # disable gradients to save memory
    torch.set_grad_enabled(False)
    
    # 6) Setup Data 
    workers= max(1,args.gpu *4) # number of workers is 4x the num GPUs but at least 1
    if platform.system()=='Windows': workers=0 # Windows 10 has bug preventing multiprocessing creating Broken Pipe: https://github.com/pytorch/pytorch/issues/2341   and  https://discourse.pymc.io/t/multiprocessing-windows-10-brokenpipeerror-errno-32-broken-pipe/2259/9
    if torch.cuda.is_available() and args.gpu > 0 and args.altModel != danQ and args.altModel != TBiNet: # danQ/TBiNet are RNNs which will NOT use multiGPUs, so scaling up the batch size will just cause VRAM OOM 
        args.batch_size = args.batch_size *args.gpu # scale up the batch size for the number of GPUs we will use

    # create Datasets ( create 2x, once for effect and once for no-effect runs)
    testData   = SeqBinDataset(y_test, test_labels_dict, test_chroms, args.seqLength, augment_shift =False, augment_rc = False, castToTensor = True, half = args.half ,motifChroms = test_motifs, motifRange = args.motifRange, protectedRange = args.protectedRange, weights = test_weights, insertA1 = args.insertA1)
    testData_RC = SeqBinDataset(y_test, test_labels_dict, test_chroms, args.seqLength, augment_shift =False, augment_rc = False, castToTensor = True, half = args.half, force_rc = args.reverseCompliment,motifChroms = test_motifs, motifRange = args.motifRange, protectedRange = args.protectedRange, weights = test_weights, insertA1 = args.insertA1)


    if args.standardiseInput != "":
        print("loading training data stats"); 
        data_means, data_sds = loadDataStats(args, testData) 
        testData.data_means = data_means ; testData.data_sds = data_sds
        testData_RC.data_means = data_means ; testData_RC.data_sds = data_sds


    # create loaders
    testLoader   = torch.utils.data.DataLoader(testData  , batch_size = args.batch_size, shuffle = False,pin_memory = args.gpu > 0 ,num_workers = workers, drop_last= False) # do NOT drop last, and do NOT shuffle as we want the original sequence in the same order
    testLoader_RC = torch.utils.data.DataLoader(testData_RC, batch_size = args.batch_size, shuffle = False,pin_memory = args.gpu > 0 ,num_workers = workers, drop_last= False) # do NOT drop last, and do NOT shuffle as we want the original sequence in the same order




    ###########################################################################
    # II) Model load: find or load the best hyper param settings and build a model
    ##########

                     
    # 2a) Load and evaluate nonlinear Model
    loadAndEvaluateModel(args,device,test_labels_dict, testLoader,testLoader_RC  , y_test, targets, y_maxabs, postfix="")


def loadAndEvaluateModel(args,device,test_labels_dict, testLoader, testLoader_RC, y_test, targets, y_maxabs = None, linear = "", postfix="") :
    # 1) create model of the hyper params defined architecture
   # if linear == "_linear" : args.hidAct = 0
    model = build_model(args, device)
    
    # 2) load the model Weights
    modelWeightsToload = args.loadWeights + linear
    if torch.cuda.is_available() and args.gpu > 0 : 
        
        if args.device != -1 : # if a specific device Id was requested, we try to map weights there
            model.load_state_dict(torch.load(modelWeightsToload,  map_location="cuda:"+str(args.device)))
        else :model.load_state_dict(torch.load(modelWeightsToload)) # otherwise attempt reconstruct the model to same devices as they were trained on, IE for multi GPUs
        
    else :  model.load_state_dict(torch.load(modelWeightsToload,  map_location='cpu'))
    #model.float() # upconvert model to float32, if it wasn't already
    
    ###########################################################################
    # III) Evaluate model:
    ######################
    print("(III) Evaluate Model:", flush = True)

    # 3) produce predictions for each sequence on the same scale as the y
    yhats_all, b_labels_all, weights, indices  = evaluateModel(args, model,device, testLoader, y_maxabs)
    yhats_all_RC, b_labels_all_RC, weights_RC, indices_RC  = evaluateModel(args, model,device, testLoader_RC, y_maxabs)


    if pheno_is_binary is False:
        # write prediction to disk
        writeSortedYs(args.out+"_y", b_labels_all, indices)
        writeSortedYs(args.out+"_yhat", yhats_all, indices)
        writeSortedYs(args.out+"_weights", weights, indices)
    
        writeSortedPredictorData(args.out+"_predictors", testLoader.dataset.labels_dict, indices)
        
        writeSortedYs(args.out+"_y_RC", b_labels_all_RC, indices_RC)
        writeSortedYs(args.out+"_yhat_RC", yhats_all_RC, indices_RC)
    
        overall_max_R_F, overall_max_S_F, overall_max_r_F, overall_mean_R_F, overall_mean_S_F, overall_mean_S_p_F, weighted_R_mean_F, weighted_S_mean_F, y_Rs_F, y_Ss_F, y_S_ps_F, y_rs_F, y_r_ps_F, overall_mean_r_F, overall_mean_r_p_F, weighted_r_mean_F, sumPP = getAccuracy(b_labels_all,yhats_all, weights)
        
        overall_max_R_RC, overall_max_S_RC, overall_max_r_RC, overall_mean_R_RC, overall_mean_S_RC, overall_mean_S_p_RC, weighted_R_mean_RC, weighted_S_mean_RC, y_Rs_RC, y_Ss_RC, y_S_ps_RC, y_rs_RC, y_r_ps_RC, overall_mean_r_RC, overall_mean_r_p_RC, weighted_r_mean_RC, sumPP = getAccuracy(b_labels_all_RC,yhats_all_RC, weights_RC)
        
        writeAccuracy(args.out+"_F",overall_max_R_F, overall_max_S_F, overall_max_r_F, overall_mean_R_F, overall_mean_S_F, overall_mean_S_p_F, weighted_R_mean_F, weighted_S_mean_F, y_Rs_F, y_Ss_F, y_S_ps_F, y_rs_F, y_r_ps_F, overall_mean_r_F, overall_mean_r_p_F, weighted_r_mean_F , sumPP,targets = targets)
        writeAccuracy(args.out+"_RC",overall_max_R_RC, overall_max_S_RC, overall_max_r_RC, overall_mean_R_RC, overall_mean_S_RC, overall_mean_S_p_RC, weighted_R_mean_RC, weighted_S_mean_RC, y_Rs_RC, y_Ss_RC, y_S_ps_RC, y_rs_RC, y_r_ps_RC, overall_mean_r_RC, overall_mean_r_p_RC, weighted_r_mean_RC , sumPP,targets = targets)
      

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

        
        writeAccuracy(args.out,overall_max_R, overall_max_S, overall_max_r,overall_mean_R, overall_mean_S, overall_mean_S_p, weighted_R_mean, weighted_S_mean, y_Rs, y_Ss, y_S_ps, y_rs, y_r_ps, overall_mean_r, overall_mean_r_p, weighted_r_mean, sumPP ,targets = targets)
        

        
        print("written results to:", args.out)
        
        testLoader.dataset.augment_rc = True
        yhats_all, b_labels_all, weights,indices  = evaluateModel(args, model,device, testLoader, y_maxabs)
        overall_max_R, overall_max_S, overall_max_r, overall_mean_R, overall_mean_S, overall_mean_S_p, weighted_R_mean, weighted_S_mean, y_Rs, y_Ss, y_S_ps, y_rs, y_r_ps, overall_mean_r, overall_mean_r_p, weighted_r_mean, sumPP = getAccuracy(b_labels_all,yhats_all, weights) 

        writeAccuracy(args.out + "_mixed",overall_max_R, overall_max_S, overall_max_r,overall_mean_R, overall_mean_S, overall_mean_S_p, weighted_R_mean, weighted_S_mean, y_Rs, y_Ss, y_S_ps, y_rs, y_r_ps, overall_mean_r, overall_mean_r_p, weighted_r_mean , sumPP,targets = targets)
       


    
            
        
     
    else :
        aurocs, auprs  = get_aurocs_and_auprs(yhats_all, b_labels_all)
        aurocs_RC, auprs_RC  = get_aurocs_and_auprs(yhats_all_RC, b_labels_all_RC)
        
        # average accuracies between forward and backward strands
        aurocs_avg = (aurocs +  aurocs_RC) / 2.
        auprs_avg = (auprs +  auprs_RC) / 2.
        
        overall_mean_auroc = np.nanmean(aurocs_avg)
        overall_mean_aupr = np.nanmean(auprs_avg)
    
        print("Averaged AUROC:",overall_mean_auroc)
        print("Averaged AUPR:", overall_mean_aupr)
        ###########################################################################
        # IV) Output results:
        #########################
        print("(IV) Output", flush = True)
    
        # 1) write out the forward/reverse breakdowns for AUC and AUPR
        fileName = args.out + linear + postfix +"all.roc_f_rc"
        with open(fileName, "w") as file:
            file.write("class" + "\t" + "auc"+ "\t" + "auc_rc" + "\t" + "auc_mean"   + "\n")
            for i in range(aurocs_avg.shape[0]) :
                file.write(targets[i]  + "\t" + str(aurocs[i]) + "\t" +  str(aurocs_RC[i] ) + "\t" +  str(aurocs_avg[i] ) +  "\n") 
    
        fileName = args.out + linear + postfix +"all.pr_f_rc"
        with open(fileName, "w") as file:
            file.write("class" + "\t" +"pr"+ "\t" + "pr_rc" + "\t" + "pr_mean"   + "\n")
            for i in range(aurocs_avg.shape[0]) :
                file.write(targets[i]  + "\t" + str(auprs[i]) + "\t" +  str(auprs_RC[i] ) + "\t" +  str(auprs_avg[i] ) +  "\n")  
    
        # 2) write out overall accuracies
        fileName = args.out + linear + postfix +"all.acc"
        with open(fileName, "w") as file:      
            file.write("Overall AUROC" +"\t" + str(overall_mean_auroc)  + "\n")      
            file.write("Overall AUPR" +"\t" + str(overall_mean_aupr)  + "\n")    
        
        print("written results to:", fileName)
  

def loadDataStats(args, trainingData): # util to load pre-calculated data means
    print("loading existing data stats")
    data_means = loadMatrixFromDisk(args.standardiseInput+"_data_means"); data_sds = loadMatrixFromDisk(args.standardiseInput+"_data_sds");

    
    return(data_means, data_sds)
        

###############################################################################
# Helper utils for accuracy calculation
###############################################################################  




def get_auroc(preds, obs):
    fpr, tpr, thresholds  = metrics.roc_curve(obs, preds, drop_intermediate=False)
    auroc = metrics.auc(fpr,tpr)
    return auroc


def get_aupr(preds, obs):
    precision, recall, thresholds  = metrics.precision_recall_curve(obs, preds)
    aupr = metrics.auc(recall,precision)
    return aupr


def get_aurocs_and_auprs(tpreds, tobs):
    tpreds_df = pd.DataFrame(tpreds)
    tobs_df = pd.DataFrame(tobs)
    
    task_list = []
    auroc_list = []
    aupr_list = []
    for task in tpreds_df:
        pred = tpreds_df[task]
        obs = tobs_df[task]
        auroc=round(get_auroc(pred,obs),5)
        aupr = round(get_aupr(pred,obs),5)
        task_list.append(task)
        auroc_list.append(auroc)
        aupr_list.append(aupr)
    return np.array(auroc_list, dtype=np.float32), np.array(aupr_list, dtype=np.float32)

###############################################################################
# Helper utils for accuracy calculation END
###############################################################################  


##################################################

# out="C:/softwares/Cluster/GIANT/miniPRS/dlpred/deepsea/results/"
# loadWeights="C:/softwares/Cluster/GIANT/miniPRS/dlpred/deepsea/results/model"     
#targetNamesloc = 'C:/softwares/Cluster/GIANT/miniPRS/dlpred/deepsea/labels/predictor.names_TF'   
  
# labels='C:/softwares/Cluster/GIANT/miniPRS/dlpred/deepsea/labels/'
# sequenceLoc='C:/softwares/Cluster/GIANT/miniPRS/dlpred/ondemand/chroms/'
# testSet='C:/softwares/Cluster/GIANT/miniPRS/dlpred/deepsea/folds/test_0'
#seed=42
#seqLength='1000'
#reverseCompliment=True

#args = parser.parse_args([ '--targetNamesloc', targetNamesloc, '--loadWeights', loadWeights,'--altModel' , '4', '--out', out, '--labels',labels, '--sequenceLoc', sequenceLoc, '--testSet', testSet, '--seqLength', seqLength])

#
def runMain(args) :
    args.bnorm = args.bnorm == 1
    args.half = args.half == 1
    args.reverseCompliment = args.reverseCompliment == 1
    args.insertA1 = args.insertA1 == 1
    
    
    print('Inference DLpred on test set',  args.testSet, "/ sequences loaded from",  args.sequenceLoc, " / labels:",args.labels,  "/ reverseCompliment:", args.reverseCompliment)    
    process(args)
    


if __name__ == '__main__':   
    parser = argparse.ArgumentParser()

    parser.add_argument("--out",required=True, help='an output location is always required')
    parser.add_argument("--seqLength",required=False, help='the length of the sequences', default=1000, type=int)  
    parser.add_argument("--sequenceLoc",required=True, help='location of sequence chrom, in files in .sbin and .sid format ')  
    parser.add_argument("--labels",required=True, help='location of the per chrom labels in .label format')  
    parser.add_argument("--seed",required=False, help='random seed for dividing the chroms', default=42, type=int)  
    parser.add_argument("--gpu",required=False, help='the number of gpus to be used. 0 for cpu.', default=0, type=int)        # 
    parser.add_argument("--targetNamesloc",required=False, help='the location of the target class names')  
    parser.add_argument("--standardiseInput",required=False, help='if input should be standardised from a location or not, if not specified', default="")   


    parser.add_argument("--widthReductionRate", default=1, help='The rate at which the network "thins" IE if we start at 1000 neurons in layer 1, then at rate of 1 (default), we half it every layer, with a rate of 2, it will half every second layer Ie we will get two layers with 1000 units each, and then two 500 units etc', type=int) 

    parser.add_argument("--bnorm",required=False, help='if batchnorm (1, default) or group norm is to be used ', default=1, type=int)   
    parser.add_argument("--device",required=False, help='the GPU device, default -1, meaning that it will try to map to original devices', default=-1, type=int)   
    parser.add_argument("--half",required=False, help='if FP16 should be used (default no)', default=0, type=int)   

    parser.add_argument("--loadWeights",required=True, help='the location of where the trained weights will be loaded from') # where we wnt to save weights

    parser.add_argument("--batch_size",required=False, help='the size of the minibatches, default :64', default=64, type=int)        # 

    

    parser.add_argument("--altModel",required=False, help='if the DanQ/Basset/DEEPSea, Basset Linear or NNP model is to be used, (for 1,2,3, 4,5 or 6) , default= 3  (= DEEPSea)', default=3, type=int)   


    parser.add_argument("--hidAct",required=False, help='the activation function, may be overriden by the best_pars', default=2, type=int)   

    parser.add_argument("--testSet",required=True, help='list of chroms for the test set')  
    
    parser.add_argument("--reverseCompliment",required=False, help='If reverse complement of the sequences are requested (default True)', default=1, type=int)  

    parser.add_argument("--motifLoc",required=False, help='folder for the motifs')  
    parser.add_argument("--protectedRange",required=False, help='location around target for which motifs cannot be inserted, default 500', default=500, type=int)  
    parser.add_argument("--motifRange",required=False, help='The left/right range around the index to look for motifs, default 1000000', default=1000000, type=int)  
    parser.add_argument("--weights",required=False, help='folder for the weights to use weighted MSE')  
 
    
    parser.add_argument("--insertA1",required=False, help='if the effect-allele should be inserted (1, default) or not (1)', default=1, type=int)       
 
    parser.set_defaults(func=runMain)
    
    
    args = parser.parse_args()
    args.func(args)

