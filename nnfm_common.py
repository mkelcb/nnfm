# -*- coding: utf-8 -*-
"""
Created on Sun May  5 18:43:39 2019

@author: mk23
"""

# DLpred common
# common functionality used by both nnfm and nnfm_test
# inc loading/standardising data and building the model
from collections import OrderedDict
import random
import numpy as np
from pathlib import Path
from types import SimpleNamespace
#from sklearn import preprocessing
import copy
from ast import literal_eval
import struct
import matplotlib
import platform
import math
if platform.system().find('Windows') == -1 :
    print("Matlab uses Agg as we are on a *nix")
    matplotlib.use('Agg')
    
from sklearn.metrics import r2_score 
from scipy.stats import pearsonr
from scipy import stats

# pytorch
import torch
import torch.nn as nn
from torch.autograd import Variable

# my own vars
from nnfmBinarizer import loadChromFromDisk, singleOneHot_reverseComplement, single_OneHot, getNumericOfNucleotide, loadAuxData, extractWindowFromSequence, get_d_code, N_value, N_
from nnfm_trainer import Conv1dSame, setModelMode # , RunningBatchNorm #, RunningBatchNorm1D

# activation functions
ReLU = 2
SELU = 6
leakyReLU=5
RReLU=7
GELU=8

# alternate models
danQ=1
Basset=2
DEEPSea=3
TBiNet=4
Basset_Linear=5
NNP_basic=6#6
DEEPSea_Linear=8#6
NNP_basic_linear=9#6

#NNP_simple=6 # 7
# data augmentation
shift_amounts = [-1,-2,-3,1,2,3] # the number of nucleotides the seq can be shifted left or right for augmentation purposes



###############################################################################
# Model construction
###############################################################################

def build_model(args,device, suppressPrint = False, mapModelToDevice = True):
    if args.altModel == danQ:  
        print("building DanQ")
        model = DanQ(sequence_length =  args.seqLength, n_genomic_features = args.n_genomic_features)
    elif args.altModel == Basset: 
        print("building Basset")
        model = build_model_Basset(sequence_length =  args.seqLength, n_genomic_features = args.n_genomic_features)
    elif args.altModel == DEEPSea: 
        print("building DEEPSea")
        model = DeepSEA(sequence_length =  args.seqLength, n_genomic_features = args.n_genomic_features)
        args.l2 = 1e-6 # overrwide the l2 to the original weight decay used by DEEPSea
        
    elif args.altModel == DEEPSea_Linear: 
        print("building DEEPSea_Linear")
        model = DEEPSea_LinearModel(sequence_length =  args.seqLength, n_genomic_features = args.n_genomic_features)
        args.l2 = 1e-6 # overrwide the l2 to the original weight decay used by DEEPSea
       
        
    elif args.altModel == Basset_Linear: 
        print("building Basset LINEAR")
        model = build_model_Basset_linear(sequence_length =  args.seqLength, n_genomic_features = args.n_genomic_features)

    elif args.altModel == NNP_basic: 
        print("building NNP_basic")
        model = NNP(sequence_length =  args.seqLength, n_genomic_features = args.n_genomic_features)

    elif args.altModel == NNP_basic_linear: 
        print("building NNP_Linear")
        model = NNP_Linear(sequence_length =  args.seqLength, n_genomic_features = args.n_genomic_features)



        
    else : 
        print("building TBiNet")
        model =TBiNet_model(sequence_length =  args.seqLength, n_genomic_features = args.n_genomic_features)

    
    # initialize the model: https://stackoverflow.com/questions/49433936/how-to-initialize-weights-in-pytorch

    if args.hidAct == SELU :  
        print("for SELU activation we use lecun normal")
        model.apply(selu_weight_init)
    else : 
        print("Glorot Uniform init")
        model.apply(glorot_weight_init)

    if args.half:  
        if suppressPrint == False : print("setting model to FP16",flush=True)
        model.half()

    if mapModelToDevice :  return(modelToDevince(model,args,device, suppressPrint = suppressPrint))
    else : return(model)

    
def modelToDevince(model,args,device, suppressPrint = False) :
    # data parallelism: https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html
    if torch.cuda.device_count() > 1 and args.gpu  > 1  : # do NOT use dataparallel GPU for inference runs as the hooks don't work: # these may not work on DataParallel models ??: https://pytorch.org/docs/0.3.1/nn.html?highlight=hook#dataparallel-layers-multi-gpu-distributed
        print("Multiple GPUs requested", args.gpu)
        gpuIDs = np.array( range(args.gpu) ).tolist() # determine the number of GPUs to use from the number supplied
        #gpuIDs = [2,3,4,5,6,7] # the first of these MUST be the 'device', IE the GPU that stores the master copy otherwise pytorch will error out
        model = nn.DataParallel(model , device_ids=gpuIDs)
        
    print(" args.gpu:",  args.gpu, "torch.cuda.device_count():", torch.cuda.device_count())
    model.to(device)
    return(model)


class DanQ(nn.Module): # adapted from: https://github.com/FunctionLab/selene/blob/master/models/danQ.py
    def __init__(self, sequence_length, n_genomic_features):
        """
        Parameters
        ----------
        sequence_length : int
            Input sequence length
        n_genomic_features : int
            Total number of features to predict
        """
        super(DanQ, self).__init__()
        self.nnet = nn.Sequential(
            nn.Conv1d(4, 320, kernel_size=26),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(
                kernel_size=13, stride=13),
            nn.Dropout(0.2))

        self.bdlstm = nn.Sequential(
            nn.LSTM(
                320, 320, num_layers=1, batch_first=True, bidirectional=True))

        self._n_channels = math.floor(
            (sequence_length - 25) / 13)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self._n_channels * 640, 925),
            nn.ReLU(inplace=True),
            nn.Linear(925, n_genomic_features),
            nn.Sigmoid())

    def forward(self, x):
        """Forward propagation of a batch.
        """
        out = self.nnet(x)
        reshape_out = out.transpose(0, 1).transpose(0, 2)
        out, _ = self.bdlstm(reshape_out)
        out = out.transpose(0, 1)
        reshape_out = out.contiguous().view(
            out.size(0), 640 * self._n_channels)
        predict = self.classifier(reshape_out)
        return predict


#TB = TBiNet_model(1000,690) ; out = TB(x2) ; out.shape
class TBiNet_model(nn.Module): # adapted from: https://github.com/dmis-lab/tbinet/blob/master/train.ipynb
    def __init__(self, sequence_length, n_genomic_features):
        """
        Parameters
        ----------
        sequence_length : int
            Input sequence length
        n_genomic_features : int
            Total number of features to predict
        """
        super(TBiNet_model, self).__init__()
        
        self.nnet = nn.Sequential(
            nn.Conv1d(4, 320, kernel_size=26),
            nn.ReLU(),
            nn.MaxPool1d(
                kernel_size=13, stride=13),
            nn.Dropout(0.2))

        self.attention = nn.Sequential(
            nn.Linear(320, 1), # the key vector, this is hx1, IE 320x1, this learns the single most important activation that each of the 320 filters produced
            nn.Softmax(dim =1) #  along axis 1, so that each of the 75 values sum to 1 
            )

        self.bdlstm = nn.Sequential(
            nn.LSTM(
                320, 320, num_layers=1, batch_first=True, bidirectional=True))
        self._n_channels = math.floor(
            (sequence_length - 25) / 13)
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self._n_channels * 640, 695),
            nn.ReLU(),
            nn.Linear(695, n_genomic_features),
            nn.Sigmoid())


    def forward(self, x):
        """Forward propagation of a batch.
        """
        # Conv operations
        C = self.nnet(x)
        
        # Attention theory:
        # h: 320, the number of filters
        # t: 75 , the maxpool activation output size
        # C: t x h, 75 x320  #  the output of the conv (or maxpool) layer
        # p: h  320 # the 'Key vector', IE a linear NN layer, with 1 neuron that 'pays attention' to the most important activations
        # a = softmax(Cp) , 75x320 x 320x1 = 75x1 # this is the output of the attention layer, finds the most important activation out of the 320 filters
        # C_scaled = C * a_tiled # Apply attention: scale each filter with the attention vector, in practice a is tiled to be the same dimension as C, and is executed as an element wise multiplication operation
   
        # Attention
        C = C.transpose(1, 2)  # swap axes so that 75x320 x 320x1 matrix multiplication works #  torch.Size([5, 75, 320]) 
        a = self.attention(C)
        a=  a.expand(-1,-1, 320) # torch.Size([5, 75, 320]) # tile the 75 attentions to cover the 320 filters
        C_scaled = C * a  # scale the Conv output by our attention vector via element wise multiplication # torch.Size([5, 75, 320])
        C_scaled =  C_scaled.transpose(1, 2)  # reset to the original axes # torch.Size([5, 320, 75]) 
        
        # LSTM
        reshape_out = C_scaled.transpose(0, 1).transpose(0, 2)
        out, _ = self.bdlstm(reshape_out)
        out = out.transpose(0, 1)
        reshape_out = out.contiguous().view(
            out.size(0), 640 * self._n_channels)
        
        # FC layers
        predict = self.classifier(reshape_out)
        return predict



class NNP_s(nn.Module):  # same as Deepsea but continuous output
    def __init__(self, sequence_length, n_genomic_features):
        """
        Parameters
        ----------
        sequence_length : int
        n_genomic_features : int
        """
        super(NNP_s, self).__init__()
        conv_kernel_size = 8
        pool_kernel_size = 4
        
        self.numLastFilters = 960 # 960 for Deep sea

        self.conv_net = nn.Sequential(
            nn.Conv1d(4, 320, kernel_size=conv_kernel_size),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(
                kernel_size=pool_kernel_size, stride=pool_kernel_size),
            #nn.Dropout(p=0.2),

            nn.Conv1d(320, 480, kernel_size=conv_kernel_size),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(
                kernel_size=pool_kernel_size, stride=pool_kernel_size),
            #nn.Dropout(p=0.2),

            nn.Conv1d(480, self.numLastFilters, kernel_size=conv_kernel_size),
            nn.ReLU(inplace=True)#,
           # nn.Dropout(p=0.5)
           )

        reduce_by = conv_kernel_size - 1
        pool_kernel_size = float(pool_kernel_size)
        self.n_channels = int(
            np.floor(
                (np.floor(
                    (sequence_length - reduce_by) / pool_kernel_size)
                 - reduce_by) / pool_kernel_size)
            - reduce_by)
        self.classifier = nn.Sequential(
            nn.Linear(self.numLastFilters * self.n_channels, n_genomic_features), # 960
            nn.ReLU(inplace=True),
            nn.Linear(n_genomic_features, n_genomic_features))

    def forward(self, x):
        """Forward propagation of a batch.
        """
        out = self.conv_net(x)
        reshape_out = out.view(out.size(0), self.numLastFilters * self.n_channels)
        predict = self.classifier(reshape_out)
        return predict



class NNP_Linear(nn.Module):  # same as NNP but linear
    def __init__(self, sequence_length, n_genomic_features):
        """
        Parameters
        ----------
        sequence_length : int
        n_genomic_features : int
        """
        super(NNP_Linear, self).__init__()
        conv_kernel_size = 8
        pool_kernel_size = 4
        
        self.numLastFilters = 960 # 960 for Deep sea

        self.conv_net = nn.Sequential(
            nn.Conv1d(4, 320, kernel_size=conv_kernel_size),
            #nn.ReLU(inplace=True),
            nn.AvgPool1d(pool_kernel_size), #nn.MaxPool1d(kernel_size=pool_kernel_size, stride=pool_kernel_size),
            nn.Dropout(p=0.2),

            nn.Conv1d(320, 480, kernel_size=conv_kernel_size),
            #nn.ReLU(inplace=True),
            nn.AvgPool1d(pool_kernel_size), #nn.MaxPool1d(kernel_size=pool_kernel_size, stride=pool_kernel_size),
            nn.Dropout(p=0.2),

            nn.Conv1d(480, self.numLastFilters, kernel_size=conv_kernel_size),
            #nn.ReLU(inplace=True),
            nn.Dropout(p=0.5))

        reduce_by = conv_kernel_size - 1
        pool_kernel_size = float(pool_kernel_size)
        self.n_channels = int(
            np.floor(
                (np.floor(
                    (sequence_length - reduce_by) / pool_kernel_size)
                 - reduce_by) / pool_kernel_size)
            - reduce_by)
        self.classifier = nn.Sequential(
            nn.Linear(self.numLastFilters * self.n_channels, n_genomic_features), # 960
            #nn.ReLU(inplace=True),
            nn.Linear(n_genomic_features, n_genomic_features),
            nn.Sigmoid()
            )

    def forward(self, x):
        """Forward propagation of a batch.
        """
        out = self.conv_net(x)
        reshape_out = out.view(out.size(0), self.numLastFilters * self.n_channels)
        predict = self.classifier(reshape_out)
        return predict



class NNP(nn.Module):  # same as Deepsea but continuous output
    def __init__(self, sequence_length, n_genomic_features):
        """
        Parameters
        ----------
        sequence_length : int
        n_genomic_features : int
        """
        super(NNP, self).__init__()
        conv_kernel_size = 8
        pool_kernel_size = 4
        
        self.numLastFilters = 960 # 960 for Deep sea

        self.conv_net = nn.Sequential(
            nn.Conv1d(4, 320, kernel_size=conv_kernel_size),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(
                kernel_size=pool_kernel_size, stride=pool_kernel_size),
            nn.Dropout(p=0.2),

            nn.Conv1d(320, 480, kernel_size=conv_kernel_size),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(
                kernel_size=pool_kernel_size, stride=pool_kernel_size),
            nn.Dropout(p=0.2),

            nn.Conv1d(480, self.numLastFilters, kernel_size=conv_kernel_size),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5))

        reduce_by = conv_kernel_size - 1
        pool_kernel_size = float(pool_kernel_size)
        self.n_channels = int(
            np.floor(
                (np.floor(
                    (sequence_length - reduce_by) / pool_kernel_size)
                 - reduce_by) / pool_kernel_size)
            - reduce_by)
        self.classifier = nn.Sequential(
            nn.Linear(self.numLastFilters * self.n_channels, n_genomic_features), # 960
            nn.ReLU(inplace=True),
            nn.Linear(n_genomic_features, n_genomic_features),
            nn.Sigmoid()
            )

    def forward(self, x):
        """Forward propagation of a batch.
        """
        out = self.conv_net(x)
        reshape_out = out.view(out.size(0), self.numLastFilters * self.n_channels)
        predict = self.classifier(reshape_out)
        return predict





class DEEPSea_LinearModel(nn.Module):  # adapted from: https://github.com/FunctionLab/selene/blob/master/models/deepsea.py
    def __init__(self, sequence_length, n_genomic_features):
        """
        Parameters
        ----------
        sequence_length : int
        n_genomic_features : int
        """
        super(DEEPSea_LinearModel, self).__init__()
        conv_kernel_size = 8
        pool_kernel_size = 4

        self.conv_net = nn.Sequential(
            nn.Conv1d(4, 320, kernel_size=conv_kernel_size),
            #nn.ReLU(inplace=True),
            nn.AvgPool1d(pool_kernel_size), # nn.MaxPool1d(kernel_size=pool_kernel_size, stride=pool_kernel_size),
            nn.Dropout(p=0.2),

            nn.Conv1d(320, 480, kernel_size=conv_kernel_size),
            #nn.ReLU(inplace=True),
            nn.AvgPool1d(pool_kernel_size),# nn.MaxPool1d(kernel_size=pool_kernel_size, stride=pool_kernel_size),
            nn.Dropout(p=0.2),

            nn.Conv1d(480, 960, kernel_size=conv_kernel_size),
            #nn.ReLU(inplace=True),
            nn.Dropout(p=0.5))

        reduce_by = conv_kernel_size - 1
        pool_kernel_size = float(pool_kernel_size)
        self.n_channels = int(
            np.floor(
                (np.floor(
                    (sequence_length - reduce_by) / pool_kernel_size)
                 - reduce_by) / pool_kernel_size)
            - reduce_by)
        self.classifier = nn.Sequential(
            nn.Linear(960 * self.n_channels, n_genomic_features),
            #nn.ReLU(inplace=True),
            nn.Linear(n_genomic_features, n_genomic_features),
            nn.Sigmoid()
            )

    def forward(self, x):
        """Forward propagation of a batch.
        """
        out = self.conv_net(x)
        reshape_out = out.view(out.size(0), 960 * self.n_channels)
        predict = self.classifier(reshape_out)
        return predict

class DeepSEA(nn.Module):  # adapted from: https://github.com/FunctionLab/selene/blob/master/models/deepsea.py
    def __init__(self, sequence_length, n_genomic_features):
        """
        Parameters
        ----------
        sequence_length : int
        n_genomic_features : int
        """
        super(DeepSEA, self).__init__()
        conv_kernel_size = 8
        pool_kernel_size = 4

        self.conv_net = nn.Sequential(
            nn.Conv1d(4, 320, kernel_size=conv_kernel_size),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(
                kernel_size=pool_kernel_size, stride=pool_kernel_size),
            nn.Dropout(p=0.2),

            nn.Conv1d(320, 480, kernel_size=conv_kernel_size),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(
                kernel_size=pool_kernel_size, stride=pool_kernel_size),
            nn.Dropout(p=0.2),

            nn.Conv1d(480, 960, kernel_size=conv_kernel_size),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5))

        reduce_by = conv_kernel_size - 1
        pool_kernel_size = float(pool_kernel_size)
        self.n_channels = int(
            np.floor(
                (np.floor(
                    (sequence_length - reduce_by) / pool_kernel_size)
                 - reduce_by) / pool_kernel_size)
            - reduce_by)
        self.classifier = nn.Sequential(
            nn.Linear(960 * self.n_channels, n_genomic_features),
            nn.ReLU(inplace=True),
            nn.Linear(n_genomic_features, n_genomic_features),
            nn.Sigmoid()
            )

    def forward(self, x):
        """Forward propagation of a batch.
        """
        out = self.conv_net(x)
        reshape_out = out.view(out.size(0), 960 * self.n_channels)
        predict = self.classifier(reshape_out)
        return predict





class LambdaBase(nn.Sequential):
    def __init__(self, fn, *args):
        super(LambdaBase, self).__init__(*args)
        self.lambda_func = fn

    def forward_prepare(self, input):
        output = []
        for module in self._modules.values():
            output.append(module(input))
        return output if output else input

class Lambda(LambdaBase):
    def forward(self, input):
        return self.lambda_func(self.forward_prepare(input))
    
    
    
    
  
def build_model_Basset_linear(sequence_length, n_genomic_features): # adapted from: https://github.com/davek44/Basset/blob/8b672320c348c6610ad79015457e0afa1c18e456/data/models/pretrained_params.txt and  https://github.com/kipoi/models/blob/master/Basset/pretrained_model_reloaded_th.py
    numFlatFeatures = findBassetFlatFeatures(sequence_length, n_genomic_features)
    model = nn.Sequential(                                   # I had to change padding to 'same' otherwise we ran out of size
            nn.Conv1d(4,300,19), # Conv1dSame(4,300,19),         #0
            nn.BatchNorm1d(300),          #1
           # nn.ReLU(),                    #2
            nn.AvgPool1d(3),            #3
            nn.Conv1d(300,200,11), # Conv1dSame(300,200,11),       #4
            nn.BatchNorm1d(200),          #5
            #nn.ReLU(),                    #6
            nn.AvgPool1d(4),            #7
            nn.Conv1d(200,200,7), # Conv1dSame(200,200,7),        #8
            nn.BatchNorm1d(200),
           # nn.ReLU(),
            nn.AvgPool1d(4),
            Lambda(lambda x: x.view(x.size(0),-1)), # Reshape,
            nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),nn.Linear(numFlatFeatures,1000)), # Linear,
            nn.BatchNorm1d(1000,1e-05,0.1,True),#BatchNorm1d,
           # nn.ReLU(),
            nn.Dropout(0.3),
            nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),nn.Linear(1000,1000)), # Linear,
            nn.BatchNorm1d(1000,1e-05,0.1,True),#BatchNorm1d,
           # nn.ReLU(),
            nn.Dropout(0.3),
            nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),nn.Linear(1000,n_genomic_features)), # Linear,
            nn.Sigmoid(),
        ) 

    return(model)
    
def build_model_Basset(sequence_length, n_genomic_features): # adapted from: https://github.com/davek44/Basset/blob/8b672320c348c6610ad79015457e0afa1c18e456/data/models/pretrained_params.txt and  https://github.com/kipoi/models/blob/master/Basset/pretrained_model_reloaded_th.py
    numFlatFeatures = findBassetFlatFeatures(sequence_length, n_genomic_features)

    model = nn.Sequential(                                   # I had to change padding to 'same' otherwise we ran out of size
            nn.Conv1d(4,300,19), # Conv1dSame(4,300,19),         #0
            nn.BatchNorm1d(300),          #1
            nn.ReLU(),                    #2
            nn.MaxPool1d(3,3),            #3
            nn.Conv1d(300,200,11), # Conv1dSame(300,200,11),       #4
            nn.BatchNorm1d(200),          #5
            nn.ReLU(),                    #6
            nn.MaxPool1d(4,4),            #7
            nn.Conv1d(200,200,7), # Conv1dSame(200,200,7),        #8
            nn.BatchNorm1d(200),
            nn.ReLU(),
            nn.MaxPool1d(4,4),
            Lambda(lambda x: x.view(x.size(0),-1)), # Reshape,
            nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),nn.Linear(numFlatFeatures,1000)), # Linear,
            nn.BatchNorm1d(1000,1e-05,0.1,True),#BatchNorm1d,
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),nn.Linear(1000,1000)), # Linear,
            nn.BatchNorm1d(1000,1e-05,0.1,True),#BatchNorm1d,
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),nn.Linear(1000,n_genomic_features)), # Linear,
            nn.Sigmoid(),
        )

    return(model)


def findBassetFlatFeatures(sequence_length, n_genomic_features) : # finds the number of flat features Basset will need depending on input seq length 
    layers = nn.Sequential(                                   # I had to change padding to 'same' otherwise we ran out of size
        nn.Conv1d(4,300,19), # Conv1dSame(4,300,19),         #0
        nn.BatchNorm1d(300),          #1
        nn.ReLU(),                    #2
        nn.MaxPool1d(3,3),            #3
        nn.Conv1d(300,200,11), # Conv1dSame(300,200,11),       #4
        nn.BatchNorm1d(200),          #5
        nn.ReLU(),                    #6
        nn.MaxPool1d(4,4),            #7
        nn.Conv1d(200,200,7), # Conv1dSame(200,200,7),        #8
        nn.BatchNorm1d(200),
        nn.ReLU(),
        nn.MaxPool1d(4,4),
        )
    batchsize = 1
    input_shape=(4,sequence_length)
    dummy_data = Variable(torch.rand(batchsize, *input_shape))
    for i in range(len(layers)) :
        dummy_data = layers[i](dummy_data)
        currentFlatFeatures = dummy_data.data.view(batchsize, -1).size(1)
    return(int(currentFlatFeatures) )

 
def addActivation(layers, hidAct): 
    if hidAct == 1 : layers.append( nn.Sigmoid() )	
    elif hidAct == ReLU : layers.append( nn.ReLU()	) #  do NOT use 'inplace=True, as that may give errors during backprop: https://discuss.pytorch.org/t/nn-relu-inplace-true-make-an-error/1109   /   https://discuss.pytorch.org/t/the-inplace-operation-of-relu/40804/9
    elif hidAct == leakyReLU : layers.append( nn.LeakyReLU(negative_slope=0.001) )	
    elif hidAct == 4 : layers.append( nn.Softplus()	)
    elif hidAct == SELU : layers.append( nn.SELU()	)
    elif hidAct == RReLU : layers.append( nn.RReLU()	) # inplace=True
    elif hidAct == GELU : layers.append( nn.GELU()	) # inplace=True 
    
    elif hidAct == 0 : layers.append( linearAct()	)
   # elif hidAct == 3 :  print("no activatioN")


class linearAct(nn.Module): # dummy linear activation layer so that the number of layers wont change between linear and nonlinear nets, as that would cause inconsistent behaviour for checkpoint_sequential of creating checkpoints at different places leading to different RAM usage
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input

class Flatten(nn.Module):
    def forward(self, input):
        #print("input hape is:", input.shape, flush=True)
        return input.view(input.size(0), -1)


def addPooling(layers,hidAct, pool_size =2 ) :
    if hidAct == 0 :layers.append( nn.AvgPool1d(pool_size) ) # linear nets get avg pooling as that is a linear operation: https://ai.stackexchange.com/questions/17937/is-a-non-linear-activation-function-needed-if-we-perform-max-pooling-after-the-c
    else : layers.append( nn.MaxPool1d(pool_size) ) # but maxpool IS non-linear, actually quite similar to ReLU
    

def addNormalization(layers,currentNumFilters,args, oneD = False) :
    if args.bnorm :  
        momentum = 0.1
        if args.accumulation_steps > 1 : momentum = 0.01 # Pytorch has the momentum in inverse, IE 01 = 0.9 in Keras. Smaller batchsizes need larger momentum IE 0.99, which in Pytorch is 0.01
        
        layers.append( nn.BatchNorm1d(currentNumFilters, eps=1e-05, momentum= momentum ) )   
       
#        if oneD : dims = (0,) # running batchnorm 1D expects a 2D input, but for FC layers this has to be 1D
#        else : dims = (0,2)
#        layers.append( RunningBatchNorm(currentNumFilters, dims = dims, halPrec = args.half ) )   

    
    
def addDropout(layers, hidAct, args, dropoutPerc = None): 
    if dropoutPerc is None : # allow to override dropout
        dropoutPerc =args.dropout
    if dropoutPerc > 0. :
        if hidAct == SELU : layers.append( nn.AlphaDropout(p=dropoutPerc)	) # use SELU for alpha dropout
        else : layers.append( nn.Dropout(p=dropoutPerc)	)
    
def findGroupNormParams(currentNumFilters, layerNorm = False) : # Group Norm needs the number of filters to be perfectly divisible by the number of groups
    if layerNorm : numGroups =1 # layernorm is when we have 1 group
    else : numGroups = int(np.ceil(np.log(currentNumFilters) ) ) # find the number of groups
    currentNumFilters = findNearestDivisible(currentNumFilters,numGroups)
    return(numGroups,currentNumFilters)
    

#    myLinear =nn.Linear(960 , 5)
#    myLinear.weight.numel()
#    myLinear.in_features
#    
#   myConv1 = nn.Conv1d(480, 960, kernel_size=5)
#   myConv1.in_features
#    myConv1.weight.data.shape[1]

#    myLinear.weight.data.shape[1]
    
    #fan_in, fan_out = init._calculate_fan_in_and_fan_out(myLinear.weight)
  #  fan_in, fan_out = init._calculate_fan_in_and_fan_out(myConv1.weight)
 #   myConv1.weight.data.shape

import torch.nn.init as init
def glorot_weight_init(m): # https://gist.github.com/jeasinema/ed9236ce743c8efaf30fa2ff732749f5
    if isinstance(m, nn.Conv1d) or isinstance(m, Conv1dSame) or isinstance(m, nn.ConvTranspose1d) or isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
        if m.bias is not None: init.constant_(m.bias, 0)
    elif  isinstance(m, nn.LSTM) :
        for name, param in m.named_parameters():
          if 'bias' in name:
             nn.init.constant_(param, 0.0)
          elif 'weight' in name:
             nn.init.xavier_uniform_(param)
        
#m = nn.LSTM(480, 960)

def selu_weight_init(m): # https://gist.github.com/jeasinema/ed9236ce743c8efaf30fa2ff732749f5
    if isinstance(m, nn.Conv1d):
        #init.kaiming_normal_(m.weight.data)
        #init.normal_(m.weight.data, 0.0, 1. / np.sqrt(m.weight.numel()))
        #init.normal_(m.weight.data, 0.0, np.sqrt( 1. / m.weight.data.shape[1]))
        fan_in, fan_out = init._calculate_fan_in_and_fan_out(m.weight)
        init.normal_(m.weight.data, 0.0, np.sqrt( 1. / fan_in))
        if m.bias is not None:
            #init.normal_(m.bias.data)      
            init.constant_(m.bias, 0)
    elif isinstance(m, Conv1dSame):
      #  init.kaiming_normal_(m.weight.data)
        #init.normal_(m.weight.data, 0.0, 1. / np.sqrt(m.weight.numel()))
        #init.normal_(m.weight.data, 0.0, np.sqrt( 1. / m.weight.data.shape[1]))
        fan_in, fan_out = init._calculate_fan_in_and_fan_out(m.weight)
        init.normal_(m.weight.data, 0.0, np.sqrt( 1. / fan_in))
        if m.bias is not None:
            #init.normal_(m.bias.data)  
            init.constant_(m.bias, 0)
            
    elif isinstance(m, nn.Conv2d):
       # init.kaiming_normal_(m.weight.data)
        #init.normal_(m.weight.data, 0.0, 1. / np.sqrt(m.weight.numel()))
        #init.normal_(m.weight.data, 0.0, np.sqrt( 1. / m.weight.data.shape[1]))
        fan_in, fan_out = init._calculate_fan_in_and_fan_out(m.weight)
        init.normal_(m.weight.data, 0.0, np.sqrt( 1. / fan_in))
        if m.bias is not None:
           # init.normal_(m.bias.data)
            init.constant_(m.bias, 0)
            
    elif isinstance(m, nn.ConvTranspose1d):
        #init.normal_(m.weight.data)
        #init.normal_(m.weight.data, 0.0, 1. / np.sqrt(m.weight.numel()))
        #init.normal_(m.weight.data, 0.0, np.sqrt( 1. / m.weight.data.shape[1]))
        fan_in, fan_out = init._calculate_fan_in_and_fan_out(m.weight)
        init.normal_(m.weight.data, 0.0, np.sqrt( 1. / fan_in))
        if m.bias is not None:
            #init.normal_(m.bias.data)
            init.constant_(m.bias, 0)
            
    elif isinstance(m, nn.Linear):
        # init.kaiming_normal_(m.weight.data)
        #init.normal_(m.bias.data)
        #init.normal_(m.weight.data, 0.0, 1. / np.sqrt(m.weight.numel()))
        #init.normal_(m.weight.data, 0.0, np.sqrt( 1. / m.weight.data.shape[1]))
        fan_in, fan_out = init._calculate_fan_in_and_fan_out(m.weight)
        init.normal_(m.weight.data, 0.0, np.sqrt( 1. / fan_in))
        init.constant_(m.bias, 0)




###############################################################################
# Model construction END
###############################################################################


###############################################################################
# Data
###############################################################################

class SeqBinDataset(torch.utils.data.Dataset):
    def __init__(self, y,labels_dict, chroms, seqLength, augment_shift = False, augment_rc = False, castToTensor = True, half = False, data_means = None, data_sds = None, origScale = False, force_rc = False, motifChroms = None, motifRange = 1000000, protectedRange = 500, weights = None, insertA1 = True):
        # Data vars
        self.y = y
        self.weights = weights
        self.labels_dict = labels_dict

        self.chroms = chroms
        self.data_means = data_means
        self.data_sds = data_sds
        self.origScale = origScale

        # Data generation vars
        self.seqLength = seqLength
        self.augment_shift = augment_shift
        self.augment_rc = augment_rc
        self.force_rc = force_rc
        self.castToTensor = castToTensor
        self.insertA1 = insertA1
        
        # motif insertion vars
        self.motifChroms = motifChroms
        self.motifRange = motifRange
        self.protectedRange = protectedRange

        # data conversion vars
        self.half = half


    def __len__(self):
        return len(self.y)

    def __getitem__(self, index): # generate one sample of data
        # I) get outcome and weight
        y_data = self.y[index]
        if self.weights is None : weight = np.ones(y_data.shape, dtype=np.float32)
        else : weight = self.weights[index]


        # II) get sequence data
        # find out chrom and bp of index SNP
        index_chr, index_bp, index_SNPid, index_A1, index_A2  =  self.labels_dict[index]
        
        # look up chrom data
        currentChrom = self.chroms[index_chr]


        # Apply optional Shift augmentation: shift the sequence by 1-3bps left or right, by  shifting the window to be extracted itself
        shift_offset =0
        if self.augment_shift and random.choice([0, 1]) == 1 :# 50/50 for doing any shifts at all 
            shift_offset = random.choice(shift_amounts) # if we DO stuff, it is either -1,-2,-3 or +1,+2,+3 
     
        # extract seq length  window sized sequence context
        buffer_half = self.seqLength // 2  # buffer_half=512
        start = index_bp + shift_offset -buffer_half
        end = index_bp + shift_offset +buffer_half -1 # -1 to exclude the last base, otherwise we would get 129 instead of 128
        b_data = extractWindowFromSequence(currentChrom,start,end, padding = True) # because this extracts everything offset by -1, the 'bp' positions will actually refer to the correct indices, despite that those were 1 based originally


        # optionally insert Motifs if this was requested
        if self.motifChroms is not None :
            insertMotifContextes(b_data, buffer_half, index_bp, index_chr, currentChrom, self.motifChroms, self.motifRange, self.protectedRange)

        # convert to one hot and float32 at the last minute to preserve RAM (as a numeric seq of int uses 16x less RAM than a 1-hot of floats)
        b_data =  single_OneHot(b_data).astype('float32')
        
        
        # 3) insert index SNPs allele: (do this last so that it overrides everything)
        if self.insertA1: 
            within_window_index_bp = index_bp - start # this is 0 based # +1 would be 1 based indices for now
            b_data[:,within_window_index_bp] = generateOneHotAlleles(index_A1,index_A2) # overwrite the alleles at the given site


        # standardise data if we have sumstats AND we are not requesting original scale data
        if self.data_means is not None and self.origScale == False:
            b_data = (b_data-self.data_means)/self.data_sds

        # 4) apply optional Reverse Complmenet augmentation
        if self.force_rc or  self.augment_rc and random.choice([0, 1]) == 1: #if forced, or if enabled randomly, then RC every other data
            b_data = singleOneHot_reverseComplement(b_data) # have to apply this BEFORE one-hot, as that is faster to reverse a 2D seq than a 3D

 
        # 5) cast to torch
        if self.castToTensor :  
            b_data = torch.from_numpy(b_data )
            y_data=  torch.tensor(y_data.astype('float32'))
            weight=  torch.tensor(weight)
            
            if self.half : 
                b_data = b_data.half()
                y_data = y_data.half()
                weight = weight.half()

        return (b_data, y_data, index, weight) # globalSNPIndex that may be used for looking up the original SNP label and data for debugging purposes



def obtainMotifsInRange(index_bp,index_chr,currentChrom,motifChroms, motifRange, protectedRange, left = True) : # obtains motif sequence datas left or right from a target within range
    # 1) obtain list of Motifs whose coordinates are within the range
    if left : # if we are searching to the left, then we want the motifs END to be greater than the index-range (but still less than the index' protected range)
        # NNNNN|1234|NNNNN|index_bp  <- want to know if '4' is within the range, but not too close to index
        within_window_Motifs = np.where(np.logical_and(motifChroms[index_chr][1] >= (index_bp - motifRange), motifChroms[index_chr][1] < (index_bp - protectedRange) ))  # https://stackoverflow.com/questions/13869173/numpy-find-index-of-the-elements-within-range
    else : # if we are searching to the right, then we want the motifs START to be less than the index+range (but still greater than the index' protected range)
        # index_bp|NNNNN|1234|NNNNN  <- want to know if '1' is within the range, but not too close to index
        within_window_Motifs = np.where(np.logical_and(motifChroms[index_chr][0] <= (index_bp + motifRange), motifChroms[index_chr][0] > (index_bp + protectedRange) ))  # https://stackoverflow.com/questions/13869173/numpy-find-index-of-the-elements-within-range

    # obtain the start/end coordinates from  the indices obtained above
    motif_starts = motifChroms[index_chr][0][within_window_Motifs]
    motif_ends = motifChroms[index_chr][1][within_window_Motifs]
    
    # 2) go through each motif and extract its sequence data
    totalLength = np.sum( (motif_ends +1) - motif_starts) # +1 for the ends, as we want to include the last nucleotide 
    allMotifSeqs = np.zeros(totalLength, dtype=np.int8 ) # pre allocate array of correct size
    start= 0 

    for i in range(len(motif_starts)) :
        motif = extractWindowFromSequence(currentChrom, motif_starts[i],motif_ends[i], padding = True)
        allMotifSeqs[start:start+len(motif)] = motif
        start = start+len(motif)
    return(allMotifSeqs)


# inserts left/right motif contextes in-place, into a b_data instance
def insertMotifContextes(b_data, buffer_half, index_bp, index_chr, currentChrom, motifChroms, motifRange, protectedRange) : 
   # obtain left/right motif context
   left_allMotifSeqs = obtainMotifsInRange(index_bp, index_chr, currentChrom, motifChroms, motifRange, protectedRange, left = True)
   right_allMotifSeqs = obtainMotifsInRange(index_bp, index_chr, currentChrom, motifChroms, motifRange, protectedRange, left = False)

   # insert them into the main b_data
   # clip left/right motif contextes to maximum length: buffer_half - protectedRange
   motifContextMax = buffer_half - protectedRange
   # IE the motivation is, that if the motifcontext is SHORTER than max, we prioritise to preserve as much of the original context as possible
   # but if is LONGER, then we take up the full space, but prefer to loose motifContext the furthest away from the index_bp    
   
   # 1) LEFT INSERTION:
   # if the motifSeq is LESS than the max, we insert it from the leftmost, to keep the original context uninterrupted
   if len(left_allMotifSeqs) <= motifContextMax : 
       # |left| ->
       # buffer_half|protected|index_bp|protected|buffer_half
       # =
       # |left|_half|protected|index_bp|protected|buffer_half   
       b_data[0:len(left_allMotifSeqs)] = left_allMotifSeqs
       
   # if motifs are LONGER than max  allowed
   else :
       # left sequence we clip from its left side, to keep the motifs closest to the index
       # |left_allMotifSeqs| ->
       # buffer_half|protected|index_bp|protected|buffer_half
       # =
       # llMotifSeqs|protected|index_bp|protected|buffer_half
       leftClip = len(left_allMotifSeqs) - motifContextMax # get how much will be left out
       b_data[0:motifContextMax] = left_allMotifSeqs[leftClip:len(left_allMotifSeqs)] # take up the full space, but only insert the parts that fit

   # 2) RIGHT INSERTION:
   # if the motifSeq is LESS than the max, we insert it from the rightmost, to keep the original context uninterrupted
   if len(right_allMotifSeqs) <= motifContextMax : 
       #                                           <- |right|
       # buffer_half|protected|index_bp|protected|buffer_half
       # =
       # buffer_half|protected|index_bp|protected|buff|right|   
       b_data[(len(b_data) - len(right_allMotifSeqs) ):len(b_data)] = right_allMotifSeqs
       
   # if motifs are LONGER than max  allowed
   else :
       # right sequence we clip from its right side, to keep the motifs closest to the index
       #                              <- |right_allMotifSeqs|
       # buffer_half|protected|index_bp|protected|buffer_half
       # =
       # buffer_half|protected|index_bp|protected|right_allMo
       #rightClip = len(right_allMotifSeqs) - motifContextMax # get how much will be left out
       b_data[(len(b_data) - motifContextMax):len(b_data)] = right_allMotifSeqs[0: motifContextMax ] # start from the start, but only insert up until it fits


# helper for above to Standardise   training data by mean centering and dividing by SD  : http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture6.pdf
def calculateDataStats(trainingData):  # in practice I found this to work better than the per channel version
    # as we cannot load in the full dataset at once, we loop through it once and calculate running 
    runningTotal = np.zeros( (4, trainingData.seqLength) )  # want each Nucleotide, and each channel to be standardised, IE the shapes must be 4x128 
    runningTotalSQ = np.zeros( (4, trainingData.seqLength) ) 
    
    castToTensor_orig = trainingData.castToTensor    
    trainingData.castToTensor =False          # dont want pytorch tensor data for now 
    trainingData_data_means_orig = trainingData.data_means; trainingData_data_sds_orig = trainingData.data_sds
    trainingData.data_means = trainingData.data_sds = None   # need to set this to None otherwise if this was already set, it would mess up the calculations
    for i in range ( len(trainingData))     :
        currentData = trainingData.__getitem__(i)[0]
        runningTotal += currentData
        runningTotalSQ += currentData**2
        
    # calculate means for each of the 4x128 inputs: mean = sum(x)/N
    data_means = runningTotal/len(trainingData)
    
    # manually recover variance from sums: var = sum(x^2)/N - mean^2
    sumSQmean= runningTotalSQ/len(trainingData)
    data_sds = np.sqrt(sumSQmean- data_means**2) # sd = sqrt(var)
    trainingData.castToTensor = castToTensor_orig # reset this for how it was
    trainingData.data_means = trainingData_data_means_orig ; trainingData.data_sds = trainingData_data_sds_orig
 
    return(data_means.astype('float32'), data_sds.astype('float32')) 
    

def calculateDataStats_perChannel(trainingData):  # same as above but instead of returning the average sequence (4x128) it just returns the average nucleotides (4,1)
    # as we cannot load in the full dataset at once, we loop through it once and calculate running 
    runningTotal = np.zeros( 4 , dtype='float64') #standardise each channel type not each nucleotide
    runningTotalSQ = np.zeros( 4 , dtype='float64') 
    
    castToTensor_orig = trainingData.castToTensor    
    trainingData.castToTensor =False          # dont want pytorch tensor data for now 
    trainingData_data_means_orig = trainingData.data_means; trainingData_data_sds_orig = trainingData.data_sds
    trainingData.data_means = trainingData.data_sds = None   # need to set this to None otherwise if this was already set, it would mess up the calculations
    
 
    for i in range ( len(trainingData))     :
        currentData = trainingData.__getitem__(i)[0]
        runningTotal += np.sum(currentData,axis=1)
        runningTotalSQ += np.sum(currentData**2,axis=1)
        
    # calculate means for each of the 4x128 inputs: mean = sum(x)/N
    count = trainingData.seqLength *  len(trainingData)
    data_means = runningTotal/count
   # runningTotal.dtype
    # manually recover variance from sums: var = sum(x^2)/N - mean^2
    data_sds = np.sqrt(runningTotalSQ/count - data_means**2) # sd = sqrt(var)
    
    # cast back to float32, and reshape these so they will work as matrix operations
    data_means= data_means.reshape(-1,1).astype('float32')
    data_sds= data_sds.reshape(-1,1).astype('float32')

    trainingData.castToTensor = castToTensor_orig # reset this for how it was
    trainingData.data_means = trainingData_data_means_orig ; trainingData.data_sds = trainingData_data_sds_orig
 
    return(data_means, data_sds) 


    
###############################################################################
# Accuracy
###############################################################################

def evaluateModel(args,model,device, testLoader, y_maxabs = None) : # generates a prediction with SD for each 
    # 1) create predicted outcomes of the right shape
    yhats_saved = [] ;  b_labels_saved = [];  b_weights_saved = [];  b_indices_saved = []

    # 2) put model into inference mode
    model_training_orig = model.training
    setModelMode(model, False) # set model to evaluation mode as usual

    y_seen=0
    for b_data, b_labels, b_index, b_weights in testLoader: # go through all batches in order
     
        y_seen+=len(b_labels)
       # print("loading batch",counter)
        
        b_data = b_data.to(device) # , augment_shift = augment_shift, augment_rc = augment_rc
        b_labels =  b_labels.to(device)
        b_index =  b_index.to(device)
        
        yhat = model(b_data)
        yhats_saved.append(yhat.detach().cpu().numpy()) 
        b_labels_saved.append(b_labels.detach().cpu().numpy()) 
        b_weights_saved.append(b_weights.detach().cpu().numpy()) 
        b_indices_saved.append(b_index.detach().cpu().numpy()) 
        
    yhats_all = np.concatenate(yhats_saved)
    yhats_all = yhats_all.astype('float32')
    
    b_labels_all = np.concatenate(b_labels_saved)  
    b_labels_all = b_labels_all.astype('float32')
    
    b_weights_all = np.concatenate(b_weights_saved)  
    b_weights_all = b_weights_all.astype('float32')
    
    b_indices_all = np.concatenate(b_indices_saved)  
    b_indices_all = b_indices_all.astype('int')
    
    # restore predictions to original scale (if this was requested)
    if y_maxabs is not None :
        yhats_all = reverse_maxAbs(yhats_all,y_maxabs)
        b_labels_all = reverse_maxAbs(b_labels_all,y_maxabs)

    # 7) reset model to what it was before
    setModelMode(model, model_training_orig)

    return(yhats_all, b_labels_all, b_weights_all, b_indices_all)


def getSumPP(y) :
    y_weights = []
    for i in range(y.shape[1]) : # as it is multitask, we want to calculate an accuracy for each pheno separately
        print(i)
        # want to weight each accuracy measure by the number of non-zero observations
        #y_weights.append( np.sum(y[:,i] != 0) ) # actually want to weight with the sum of PPs, as that gives better idea of the total signal
        y_weights.append( np.sum(y[:,i] ) )
    return(y_weights)
       
 
def getAccuracy(y,yhat, weights) : # weights here is the overall weights to the y
    sumPP = getSumPP(y)
  
    y_weights = sumPP / np.sum(sumPP)

    y_Rs = []
    y_Ss = []
    y_S_ps = []
    y_rs = []
    y_r_ps = []
    for i in range(y.shape[1]) : # as it is multitask, we want to calculate an accuracy for each pheno separately
        Rsq,S, r, r_p = getAccuracyMetrics(yhat[:,i],y[:,i], weights[:,i])
        y_Rs.append(Rsq)
        y_Ss.append(S.correlation ) # take absolute value as we are interested in its predictive power, not the direction
        y_S_ps.append(S.pvalue)
        y_rs.append(r ) # take absolute value as we are interested in its predictive power, not the direction
        y_r_ps.append(r_p)
  
    # create overall mean
    overall_mean_R = np.nanmean(y_Rs)
    overall_mean_S = np.nanmean(np.abs(y_Ss))
    overall_mean_S_p = np.median(y_S_ps) # median
    
    overall_mean_r = np.nanmean(np.abs(y_rs))
    overall_mean_r_p = np.median(y_r_ps)  # median
 
 
    
    # overal max
    overall_max_R = np.nanmax(y_Rs)
    overall_max_S = np.nanmax(np.abs(y_Ss))
    overall_max_r = np.nanmax(np.abs(y_rs))
    
    # get the weighted versions too
    weighted_R_mean = np.nansum(y_Rs * y_weights)
    weighted_S_mean = np.nansum(np.abs(y_Ss) * y_weights)
    weighted_r_mean = np.nansum(np.abs(y_rs) * y_weights)
    
    return(overall_max_R, overall_max_S, overall_max_r, overall_mean_R, overall_mean_S, overall_mean_S_p, weighted_R_mean, weighted_S_mean, y_Rs, y_Ss, y_S_ps, y_rs, y_r_ps, overall_mean_r, overall_mean_r_p, weighted_r_mean, sumPP)
    

# calculates R^2, Spearmans corr and peaersons corr, all using weights 
def getAccuracyMetrics(yhats_saved,b_labels_saved, b_weights_saved) :
    if type(yhats_saved) is list :
        yhats_all = np.concatenate(yhats_saved)
        b_labels_all = np.concatenate(b_labels_saved)
        b_weights_all = np.concatenate(b_weights_saved)
    else :
        yhats_all = yhats_saved
        b_labels_all = b_labels_saved
        b_weights_all = b_weights_saved
        
    b_labels_all = b_labels_all.astype('float32')
    yhats_all = yhats_all.astype('float32')
    b_weights_all = b_weights_all.astype('float32')
    
    # use the weights as a mask to extract to only calculate accuracy among the non-zero weights
    mask_indices = np.where(b_weights_all != 0)
    yhats_all = yhats_all[mask_indices]
    b_labels_all = b_labels_all[mask_indices]
    
    # replace Nans  with the (no Nan) mean of the yhats
    meanYhat = np.nanmean(yhats_all) ;  meanYhat = np.nan_to_num(meanYhat)   
    yhats_all = np.nan_to_num(yhats_all, nan=meanYhat)   

    # get R^2, Spearman and Pearson correlation 
    RsQ = r2_score(b_labels_all, yhats_all) 
    spearmans_Corr = stats.spearmanr(b_labels_all, yhats_all)
    r, r_p = pearsonr(b_labels_all, yhats_all)

    return(RsQ,spearmans_Corr, r, r_p)



def writeAccuracy(out,overall_max_R, overall_max_S, overall_max_r,overall_mean_R, overall_mean_S, overall_mean_S_p, weighted_R_mean, weighted_S_mean, y_Rs, y_Ss, y_S_ps, y_rs, y_r_ps, overall_mean_r, overall_mean_r_p, weighted_r_mean, sumPP ,targets = None) :
    with open(out + "model.acc", "w") as file:    
        file.write("overall_mean_R:\t" + str(overall_mean_R) + "\n" ) 
        file.write("weighted_R_mean:\t" + str(weighted_R_mean) + "\n" ) 
        file.write("overall_mean_abs_S (median p):\t" + str(overall_mean_S) + "(" +str(overall_mean_S_p)+ ")" + "\n" ) 
        file.write("weighted_mean_abs_S:\t" + str(weighted_S_mean) + "\n" ) 
        file.write("overall_mean_abs_r (median p):\t" + str(overall_mean_r) + "(" +str(overall_mean_r_p)+ ")" + "\n" ) 
        file.write("weighted_mean_abs_r:\t" + str(weighted_r_mean) + "\n" ) 
        file.write("max_R:\t" + str(overall_max_R) + "\n" ) 
        file.write("overall_max_S:\t" + str(overall_max_S) + "\n" ) 
        file.write("overall_max_r:\t" + str(overall_max_r) + "\n" ) 
        
        
        
        
    with open(out + "model_breakdown.acc", "w") as file:  
        if targets is not None : line = "trait\tR^2\tS\tS_p\tr\tr_p\tsumPP"
        else : line="R^2\tS\tS_p\tr\tr_p\tsumPP"
        file.write(line + "\n" ) 
        
        for i in range(len(y_Ss)) :
            if targets is not None : line= targets[i] +"\t" + str(y_Rs[i]) +"\t" + str(y_Ss[i]) +"\t" + str(y_S_ps[i]) +"\t" + str(y_rs[i]) +"\t" + str(y_r_ps[i]) +"\t" + str(sumPP[i])
            else : line =  str(y_Rs[i]) +"\t" + str(y_Ss[i]) +"\t" + str(y_S_ps[i]) +"\t" + str(y_rs[i]) +"\t" + str(y_r_ps[i]) +"\t" + str(sumPP[i])
            file.write(line + "\n" ) 
            
            
def writeSortedYs(out, y, indices) : # writes out y/yhats to disk, in the original index order
    sorted_indices = np.argsort(indices) # sort the indices 
    with open(out, "w") as file: 
        for i in range(len(sorted_indices)) :
            i_index = sorted_indices[i]

            num_representation_i = y[i_index]
            text_representation = [str(num) for num in num_representation_i]
            y_text = ','.join(text_representation) # convert array to string
            file.write(str(i) +"," + y_text + "\n" )         
        
def writeSortedPredictorData(out, labels_dict, indices) : # writes out y/yhats to disk, in the original index order
    sorted_indices = np.argsort(indices) # sort the indices 
    with open(out, "w") as file: 
        for i in range(len(sorted_indices)) :
            i_index = sorted_indices[i]

            index_chr, index_bp, index_SNPid, index_A1, index_A2 = labels_dict[i_index]

            predictorInfos = str(index_chr)+","+str(index_bp) + "," + str(index_SNPid)
            file.write(str(i) +"," + predictorInfos + "\n" )         
         


###############################################################################
# Utils
###############################################################################


def findNearestDivisible(value,base = 2.) : # finds the nearest value divisible by a given number
    return( int( base * np.floor( value / base )) )


def generateOneHotAlleles(A1,A2,A1_prob = 1.) : # generates a 1 hot-type nucleotide with the appropriate probabilities
    if A1 == N_ : return(np.ones( (4) ,dtype=np.float32) * N_value) # edge case if this is a missing call nucleotide
    else :
        nucleotide = np.zeros( (4) ,dtype=np.float32 )
        nucleotide[A1] = A1_prob
        nucleotide[A2] = 1.- A1_prob
        return(nucleotide)

def y_origscale(y,y_mean, y_sd): # transforms y back to raw scale
    
    y_rawscale = y *y_sd + y_mean 
    #np.mean(y_rawscale) # 2.0
    #np.var(y_rawscale) # 4.0
    return (y_rawscale)


def y_standardise(y,y_mean, y_sd):  # transforms a raw y into zscore
    y_standardised = (y - y_mean) / y_sd 
    #np.mean(y_standardised) # 2.0
    #np.var(y_standardised) # 4.0
    return (y_standardised)


# Sum over an axis is a reduction operation so the specified axis disappears.  
def zscore(a, axis=0, EPSILON = -1):
    a = a.astype('float32')  # if we don't cast this then this would upconvert everything to float64
    mns = a.mean(axis=axis)
    sstd = a.std(axis=axis)
    mns = mns.astype('float32')
    sstd = sstd.astype('float32')
    if EPSILON != -1 : sstd += EPSILON # for numerical stability
    sstd[sstd==0] = 1 # dont want division by zero
    #a = (a - mns) / sstd
    a -= mns
    a /= sstd
    if len(mns) == 1 : # if there is only one element, dont return an array, just the scalar
        mns = mns[0]
        sstd = sstd[0]
    return a, mns, sstd


# for sparse data, like multitask learning, Maxabs is better than zscore:
# https://towardsdatascience.com/preprocessing-with-sklearn-a-complete-and-comprehensive-guide-670cb98fcfb9
# https://www.programmersought.com/article/6445180812/
def maxAbs(a, axis=0):
    a_maxabs = np.max(np.abs(a), axis=axis)
    a_scaled = a / a_maxabs
    
    return(a_scaled, a_maxabs)

def y_standardise_maxAbs(a, a_maxabs):
    a_scaled = a / a_maxabs
    return(a_scaled)

def reverse_maxAbs(a_scaled,a_maxabs):
    a_reversed = a_scaled * a_maxabs
    return(a_reversed)




###############################################################################
# Saving / Loading
###############################################################################
def loadTargetNames(targetNamesloc) :
    targets = []
    with open(targetNamesloc, "r") as id:
        for i in id:
            itmp = i.rstrip().split("\t")
            targets.append(itmp[0])
    return(targets)


def loadAllMotifData(usedChroms,location) : # loads all Motifs
    chromCounter= 0
    motifChroms = {} # keyed by chrom strores starts/ends of all motifChroms[chromnum] = [starts] , [ends]
    for currentChrom, value in usedChroms.items(): # go through all used chromosomes
   
        motifLoc = location+str(currentChrom)+".motif"
        motifChroms[currentChrom] = loadChromMotifData(motifLoc)
        chromCounter+=1     
        print("motifs loaded:",chromCounter, "/",len(usedChroms), end='\r')
    return(motifChroms)
     

#location = motifLoc
def loadChromMotifData(location) :
    chromMotif_starts = []
    chromMotif_ends = []
    with open(location, "r") as id:
         for i in id:
             itmp = i.rstrip().split() 
             chromMotif_starts.append( int(itmp[1]) )
             chromMotif_ends.append( int(itmp[2]) )
    chromMotif_starts = np.asarray(chromMotif_starts)
    chromMotif_ends = np.asarray(chromMotif_ends)
    return(chromMotif_starts,chromMotif_ends)



def loadChroms(location, seqLoc) :
    chrom_list = []
    with open(location, "r") as id:
        for i in id:
            itmp = i.rstrip().split()
            chrom_list.append(int(itmp[0]) )
            
    chroms = OrderedDict()
    for i in range(len(chrom_list)) :
        CHR= chrom_list[i]
        # load the binary in
        chroms[CHR] = loadChromFromDisk(seqLoc + str(CHR) )
        # as this takes a long time, want to update a counter how many are left to load
        print("chrom loaded:",i+1, "/",len(chrom_list), end='\r')
    print("all sequence data loaded", flush = True)
    return(chroms)
    
    
# signature of file
#index,SNPSid,label,group
#2,rs28544273,-0.000170077091933686,control
# usedChroms = training_chroms
#auxDataLocation = args.auxDataLocation
#location = args.labels
#shuffleLabels = False
#currentChrom =21
def loadLabels(location, usedChroms, shuffleLabels = False, extension = ".label") : # loads all labels from the used chroms
    betas = [] # an overall list of all betas
    labels_dict = [] # a dictionary that stores: [globalIndex] = [snp_chr, bp, SNPid, A1, A2]
    # indices match for the betas and labels_dict
    chromCounter= 0
    for currentChrom, value in usedChroms.items(): # go through all used chromosomes
        
        # 1) for chrom: load .labels: signature: chr, id, bp,  y1,y2, ... ,y919
        with open(location+str(currentChrom)+ extension, "r") as id:
            for i in id:
                itmp = i.rstrip().split(",") 
                snp_chr = int(itmp[0])
                SNPid = itmp[1]
                bp = int(itmp[2])
                A1 = itmp[3]
                A2 = itmp[4]

                # get the outcome
                beta = np.array([float(i) for i in itmp[5:len(itmp)] ], dtype =np.float32 )
                betas.append(beta)
                labels_dict.append( [ snp_chr, bp, SNPid, getNumericOfNucleotide(A1) , getNumericOfNucleotide(A2)]  )

        chromCounter+=1     
        print("labels loaded:",chromCounter, "/",len(usedChroms), end='\r')
    print("all labels loaded", flush = True)
    

    if shuffleLabels :
        print("shuffling labels!")
        random.shuffle(betas) # this modifies it in place. also applying it to the y outside


    betas = np.asarray(betas).astype('float32') # store as 32 bit floats
    

    return(betas, labels_dict)







#location = out  + ".ymeansd"
#outFile = location
def loadYMeanSds(location) : # attempts to load SD and mean for the y (text file assumed to have 2 lines, the mean and sd)
    y_mean = None
    y_sd = None

    with open(location, "r") as id:
        for i in id:
            itmp = i.rstrip().split(",")
            
            if y_mean is None : y_mean =  itmp
            elif  y_sd is None: y_sd = itmp
            
    y_mean = [ float(x) for x in y_mean ]
    y_mean = np.array(y_mean, dtype=np.float32)
    y_sd = [ float(x) for x in y_sd ]
    y_sd = np.array(y_sd, dtype=np.float32)
    
    return(y_mean,y_sd)
    
  
def writeYMeanSds(outFile, y_mean,y_sd ) :
    with open(outFile + "model.ymeansd", "w") as file: 
        line_means=str(y_mean[0])
        line_sds=str(y_sd[0])
        for i in range(1,len(y_mean)) :
            line_means+="," + str(y_mean[i])
            line_sds+="," + str(y_sd[i])
            
        file.write(line_means + "\n" ) 
        file.write(line_sds + "\n" ) 


def loadYMaxAbs(location) : 
    with open(location, "r") as id:
        for i in id:
            a_maxabs_loaded = i.rstrip().split(",")
    a_maxabs_loaded = [ float(x) for x in a_maxabs_loaded ]
    a_maxabs_loaded = np.array(a_maxabs_loaded, dtype=np.float32)

    return(a_maxabs_loaded)


def writeYMaxAbs(outFile, a_maxabs ) :
    with open(outFile + "model.ymaxabs", "w") as file: 
        line=str(a_maxabs[0])
        for i in range(1,len(a_maxabs)) :
            line+="," + str(a_maxabs[i])
        file.write(line + "\n" ) 



def writeKNeT_bestPars(outFile, best_pars ) :
    with open(outFile + "model.best_pars", "w") as file: 
            file.write(str(best_pars) ) 


def loadKNeT_bestPars(outFile) :
    s=""
    with open(outFile + ".best_pars", "r") as file: 
        for i in file:
            s+=i
    best_pars= literal_eval(s)        
    return(best_pars) 
    
    

# writes a matrix onto disk in a binary (2 files, 1 that stores the dimensions, the other the binary data)
def writeMatrixToDisk(location,data, dataType ="float32") :
    d_code = get_d_code(dataType)
    
    # get dimensions of matrix
    nrows = data.shape[0]
    ncols = data.shape[1]
    
    # write the dimensions onto disk
    with open(location + ".id", "w") as idFile: 
        idFile.write(str(nrows) + "\t" +str(ncols) )
        
    # flatten matrix
    flat = data.ravel()

    flatData = struct.pack(d_code*len(flat),*flat  )
    with open(location + ".bin", "wb") as flat_File: 
        flat_File.write(flatData) 
    

# loads matrix from disk ( that was written by the above)
def loadMatrixFromDisk(location, dataType ="float32") :
    d_code = get_d_code(dataType)
    
    # load id file to get dimensions
    with open(location + ".id", "r") as idFile:
        itmp = idFile.readline().rstrip().split()
        nrows = int(itmp[0])
        ncols = int(itmp[1])
        
    # how many elements to expect in the binary in total
    totalNum =nrows * ncols
    
    # open binary file
    with open(location + ".bin", "rb") as BinFile:
        BinFileContent = BinFile.read()
  
    # reformat data into correct dimensions
    flat = np.array( struct.unpack(d_code*totalNum, BinFileContent  ), dtype = dataType )
    data = flat.reshape(nrows,ncols)
    return(data)

###############################################################################
# Helpers and Diagnositic functions
###############################################################################  
    
def model_diagnostics(model, out):
    totalParams = sum([p.numel() for p in model.parameters()])
    totalParams # 726270  # 211
    resultsLines = []
    resultsLines.append("model has total num params: " + str( totalParams) )

    

    resultsLines.append("pytorch model \n")
    resultsLines.append(str(model))
    
    with open(out+"_diag", "w") as file: 
        for i in range(len(resultsLines) ):
            print(resultsLines[i])
            file.write(str(resultsLines[i])  + "\n")
            
    print("written diagnostics info to", out+"_diag")  

    
def printElapsedTime(start,end, text ="") : # https://stackoverflow.com/questions/27779677/how-to-format-elapsed-time-from-seconds-to-hours-minutes-seconds-and-milliseco
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print(text + "{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds), flush=True)   

def mergeArgsAndParams(args,params) :
    argsMerged = copy.deepcopy(args) # createa deep copy of the original args object
    argsMerged = vars(argsMerged) # convert it to dictionary, so that we can easily copy the key/value pairs

    # go through the keys params and overwrite the corresponding entry in the args  (the argnames must match)
    for key, value in params.items():
        argsMerged[key] = value

    #argsMerged = { 'no_cuda': False, 'batch_size': 64, 'test_batch_size': 1000, 'epochs': 10, 'lr':0.01, 'momentum': 0.5, 'seed': 1, 'log_interval': 10 }
    argsMerged = SimpleNamespace(**argsMerged) # convert back to namespace, as that is what the build_model expects
    return(argsMerged)   

def getSizeInMBs(myObject) :
    if myObject is None : return 0.
    return ( np.round( myObject.nbytes  / 1024/ 1024 )  )

def getSizeInGBs(myObject) :
    if myObject is None : return 0.
    return ( np.round( myObject.nbytes * 10 / 1024/ 1024 / 1024 ) / 10  )

def getSize(seqLength,num_labels, prefix = "training") :
    sizeInMBs = seqLength * num_labels  / 1024/ 1024
    if sizeInMBs < 1024 :
        print(prefix, "RAM usage will be",np.round( sizeInMBs ), "MBs")
    else :
        print(prefix, "RAM usage will be", np.round( sizeInMBs * 10 / 1024 ) / 10, "GBs")


################################################################
   