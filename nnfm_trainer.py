# -*- coding: utf-8 -*-

#MIT License

#Copyright (c) 2020 Martin Kelemen

#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all 
# copies or substantial portions of the Software.

#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.



import numpy as np


import sys 
import time

import torch
import torch.nn as nn

from scipy import stats
import torch.optim as optim
#import torch.nn.init as init




###############################################################################
# Main vars
###############################################################################


OPTIMIZER_SGD = 0
OPTIMIZER_ADAM = 1
OPTIMIZER_AMSGRAD = 2

#device = None
#NETWORK_DATATYPE = torch.float32# 'float32'
#
#def getNetworkDatatype_numpy() :
#    global NETWORK_DATATYPE
#    if NETWORK_DATATYPE == torch.float32 : return np.float32
#    else  : return np.float64

###############################################################################
# Learning
###############################################################################
#def setHalfPrecision(model) :
#    model = getModel(model)
#    model.half()  # convert to half precision
#    for layer in model.modules():
#        if isinstance(layer, nn.BatchNorm1d) or isinstance(layer, nn.GroupNorm):
#            layer.float()
    
#    torch.set_default_dtype(torch.float16)
#    torch.set_default_tensor_type(torch.float16)

class MSE_Weighted(torch.nn.Module):
    
    def __init__(self):
        super(MSE_Weighted,self).__init__()

    def forward(self,input, target, weight):
        return ( weight * (input - target)**2 ).sum() / weight.sum()
       


def learn(model, device, trainLoader, validLoader, saveModelLocation = None, epochMaxImproveThreshold = 30, learnRate = 0.1, momentum = 0.9,  gamma = 0.999, suppressPrint = False, half = False, decayRate = 0.96, accumulation_steps =6, debugOut = None, l2= 0., sgd = False, pheno_is_binary = True):
 
    model_training_orig = model.training
    setModelMode(model, True)

    # I) setup optimiser & loss function
    if half : EPSILON = 1e-4 # this is for FP32, for FP16, this sohuld be 1e-4
    else :EPSILON = 1e-7  # 1e-8 default for pytorch,  1e-7 , default for keras

    if sgd : optimizer = optim.SGD(model.parameters(), lr=learnRate, momentum=momentum, weight_decay= l2)
    else : optimizer = optim.Adam(model.parameters(), lr=learnRate, betas=(momentum, gamma), eps=EPSILON, weight_decay= l2)   
    
    
    # decide on loss function
    if pheno_is_binary : criterion = nn.BCELoss()
    else :  criterion = MSE_Weighted() # nn.MSELoss()
     # MSE_Weighted() # nn.MSELoss() # loss function for regression is MSE, regression only 1 column
          
 
    # setup results & early stop logic ( record highest validation accuracy and its epoch)
    results = {}
    results["epochs"] = list()
    results["train_loss"]  = list()
    results["valid_loss"]  = list()
    results['lowestLoss'] = None
    results['lowestLoss_epoch'] = -1

    eta = learnRate
    currentEta = eta
    if suppressPrint == False : print ("TF classifier (EPSILON = 1e-7): Training started iterations until no improvement epochs: " + str(epochMaxImproveThreshold), "/ learn Rate:", str(eta), " / exponential decay:", decayRate, " / l2: ", l2, " / pheno_is_binary: ", pheno_is_binary, flush=True  )
    t = 0
    

    training_hasntmprovedNumIts = 0
    

    if decayRate != -1 : lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)
    
    while training_hasntmprovedNumIts < epochMaxImproveThreshold : #t < epochs: #for t in range(0, epochs):
        out_str = " | epoch: "  + str(t) # "[{0:4d}] ".format(t)
        results["epochs"].append(t) # add the epoch's number
        
        # 1) Complete an entire training cycle: Forward then Backward propagation, then update weights, do this for ALL minibatches in sequence
        start_time = time.time()
        currentBatchNum = 0

        epoch_trainingLoss = 0.
        
        torch.set_grad_enabled(True) ; model.train() # enable grads for training
        if t == 0 : 
            torch.set_grad_enabled(False)
            model.eval() # for the first epoch we want to know what the baseline prediction looks like, must use .eval() otherwise BN-like layers would still learn/update their params which could cause numerical issues


       # yhats_saved = [] ;  b_labels_saved = []
        optimizer.zero_grad()
        #model.half()
        for b_data, b_labels, b_indices, b_weights in trainLoader:
            #print ("batch shape: " ,b_data.shape, " / labels shape:", b_labels.shape , flush=True  )
   
            b_data = b_data.to(device) # , augment_shift = augment_shift, augment_rc = augment_rc
            b_labels =  b_labels.to(device)
            b_weights =  b_weights.to(device)
            
            # perform full learning cycle, FP, BP and update weights
            #optimizer.zero_grad()   # zero the gradient buffers

            # Forward Propagate
            yhat = model(b_data) 
            if isinstance(criterion, nn.BCELoss) : loss = criterion(yhat, b_labels)  / accumulation_steps #, usually they recommend to normalize by this, but as I am not dividing by the accumulation steps, as I want total error not average # calculate error
            else : loss = criterion(yhat, b_labels, b_weights)  / accumulation_steps
            # calculate accuracy: must do this all at once, as if we average per minibatch, then that may give slightly optimistic results for very low numbers
          #  yhats_saved.append(yhat.view(-1).detach().cpu().numpy()) ;  b_labels_saved.append(b_labels.view(-1).detach().cpu().numpy()) ; 
       
        
            if t > 0 : # first epoch will want to evaluate loss based on no training
                loss.backward()   # Backprop
                
                # Gradient accumulation: only update params every accumulation_steps
                if (currentBatchNum+1) % accumulation_steps == 0:   
                    #torch.nn.utils.clip_grad_norm_(getModel(model).parameters(), 2) #  0.001
                    optimizer.step()                            # Now we can do an optimizer step
                    optimizer.zero_grad()                           # Reset gradients tensors

            epoch_trainingLoss+= loss.item() * (b_labels.shape[0] * b_labels.shape[1]) * accumulation_steps # get total loss # batch size: the criterion returns the AVG loss for the whole epoch, if we want to total loss, we need to multiply by batch_size: https://discuss.pytorch.org/t/mnist-dataset-why-is-my-loss-so-high-beginner/62670

            # cosmetics: update prograss bar
            barPos =  float(currentBatchNum) / len(trainLoader) # the bar position in %
            barPos = round(20 * barPos) # scale it to 1-20
            if suppressPrint == False : 
                sys.stdout.write('\r')
                sys.stdout.write("[%-20s] %d%%" % ('='*barPos, 5*barPos))
                sys.stdout.flush()  
            currentBatchNum += 1
            ###################### end of Training ########################
  
        # Evaluate Training
       # train_accuracy, train_spearmanr = getCoefficientOfDetermination(yhats_saved,b_labels_saved)
        epoch_trainingLoss= epoch_trainingLoss /len(trainLoader) # divide by number of batches to get a loss comparable between training/valid
        out_str =  out_str + " / Train loss: "+ str( round( epoch_trainingLoss,3) )   #   + " / R^2: " + str(  round( train_accuracy,4) ) + " / S: " + str(  round( train_spearmanr[0],4) ) + "(" + str(  round( train_spearmanr[1],4) )  +")"
        results["train_loss"].append(epoch_trainingLoss)  
       # results["train_acc"].append(train_accuracy)
       # results["train_S"].append(train_spearmanr[0])
       # results["train_Sp"].append(train_spearmanr[1])


        # Eval Test 
        torch.set_grad_enabled(False) ; model.eval() # disable gradients to save memory

        epoch_validLoss = 0.
        #yhats_saved = [] ;  b_labels_saved = []; 
        #model.float()
        for b_data, b_labels, b_indices, b_weights in validLoader:
  
            b_data = b_data.to(device) 
            b_labels =  b_labels.to(device)  # b_labels.view(-1,1).to(device)  # torch complains if size is (n,) instead of (n,1)
            b_weights =  b_weights.to(device)
            
            # Forward Propagate
            yhat = model(b_data) 
            if isinstance(criterion, nn.BCELoss) : loss = criterion(yhat, b_labels) # loss function
            else : loss = criterion(yhat, b_labels, b_weights)
            # calculate accuracy: must do this all at once, as if we average per minibatch, then that may give slightly optimistic results for very low numbers
            #yhats_saved.append(yhat.view(-1).detach().cpu().numpy()) ;  b_labels_saved.append(b_labels.view(-1).detach().cpu().numpy()) ; 
                  
     
            # get total loss for iteration
            epoch_validLoss+= loss.item() * (b_labels.shape[0] * b_labels.shape[1]) # batch size: the criterion returns the AVG loss for the whole epoch, if we want to total loss, we need to multiply by batch_size: https://discuss.pytorch.org/t/mnist-dataset-why-is-my-loss-so-high-beginner/62670
    
        # Evaluate Validation
       # valid_accuracy, valid_spearmanr = getCoefficientOfDetermination(yhats_saved,b_labels_saved)
        epoch_validLoss= epoch_validLoss /len(validLoader) # divide by number of batches to get a loss comparable between training/valid
        results["valid_loss"].append(epoch_validLoss)
      #  results["valid_acc"].append(valid_accuracy)
      #  results["valid_S"].append(valid_spearmanr[0])
       # results["valid_Sp"].append(valid_spearmanr[1])
        out_str =  out_str + " | Valid loss: "+  str(round(epoch_validLoss,3))# + " / R^2: " +  str(round( valid_accuracy,4)) + " / S: " +  str(  round( valid_spearmanr[0],4) )  + "(" +  str(  round( valid_spearmanr[1],4) ) + ")"
       
        # if training has improved over the best so far, reset counter
        if results['lowestLoss'] is None or epoch_validLoss < results['lowestLoss']:
            results['lowestLoss'] = epoch_validLoss
            results['lowestLoss_epoch'] = t
            training_hasntmprovedNumIts = 0 
        else :
            training_hasntmprovedNumIts += 1
        
        # save the entire model weights for this epoch
        if saveModelLocation is not None:
            torch.save(getModel(model).state_dict(), saveModelLocation + str(t))  # could alternatively store it on the CPU as:  for k, v in state_dict.items():  state_dict[k] = v.cpu()   # https://discuss.pytorch.org/t/how-to-get-a-cpu-state-dict/24712

        elapsed_time = time.time() - start_time 
        if suppressPrint == False : print(out_str + " / " + str( round(elapsed_time) ) + " secs (LR: " + str(round(currentEta,5)) + ")" , flush=True)
        
        # update learning rate
        if decayRate != -1 and t > 0 :
            lr_scheduler.step()
            currentEta = lr_scheduler.get_last_lr()[0]
        
        t += 1
         
    setModelMode(model, model_training_orig)
    return ( { "results" : results})  


###############################################################################
# Additional Layer types:
###############################################################################

def writeDebugOutput(debugOut,yhats_saved, b_labels_saved,b_weights_saved,t) :
#    yhats_all = torch.cat(yhats_saved)
#    b_labels_saved = torch.cat(b_labels_saved)
#    yhats_all = yhats_all.cpu().numpy()
#    b_labels_saved = b_labels_saved.cpu().numpy()
#    

    
    if type(yhats_saved) is list :
        yhats_all = np.concatenate(yhats_saved)
        b_labels_all = np.concatenate(b_labels_saved)
        b_weights_all = np.concatenate(b_weights_saved)
    else :
        yhats_all = yhats_saved
        b_labels_all = b_labels_saved
        b_weights_all = b_weights_saved
    with open(debugOut+ "_" + str(t), "w") as file: 
        file.write("y(" + str(np.mean(b_labels_all)) +")"  + "\t" + "yhat" + "\t"+ "weight"  + "\n")
        for i in range(len(b_labels_all)) :
            file.write( str(b_labels_all[i])  + "\t" + str(yhats_all[i]) + "\t"+ str(b_weights_all[i])  + "\n")   



# Same padding issue in Pytorch:
# this works for ODD sized filters only, as for even sized filters, the 'same' padding would be a fraction , eg 9.5
# No existing solution for this in pytorch yet, as it would require assymetric padding
def solveSamePadding (filter_size,stride,dilation) : # 'same' padding for Pytorch: https://discuss.pytorch.org/t/how-to-keep-the-shape-of-input-and-output-same-when-dilation-conv/14338/7
    padding = (-1 * stride + (filter_size-1)*(dilation-1) + filter_size ) /2 
    return( int(padding) )
      

# a hacky solution from https://github.com/pytorch/pytorch/issues/3867
# its a wrapper for the reuglar Conv1D, and  works by removing the last element for even sized outputs 
# this is likely to be less performance efficient than would be manually to ensure non-even sizes
class Conv1dSame(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super().__init__()

        self.cut_last_element = (kernel_size % 2 == 0 and stride == 1 and dilation % 2 == 1)

        self.padding = int(np.ceil((1 - stride + dilation * (kernel_size-1))/2) )
        self.padding = tuple( [  self.padding ] )
        #print("Conv Same padding is:", self.padding, "cut_last_element: ", self.cut_last_element)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=self.padding[0], stride=stride, dilation=dilation)

        # add class properties for compatibility
        self.dilation = tuple([dilation])
        self.kernel_size = tuple([kernel_size])
        self.in_channels = in_channels
        self.out_channels = out_channels  
        self.stride = tuple([stride])
        
        
        self.weight = self.conv.weight
        self.bias = self.conv.bias


    def forward(self, x):
        print("Conv1dSame: x.shape: ", x.shape, flush= True)
        if self.cut_last_element:  return self.conv(x)[:, :, :-1]
        else:  return self.conv(x)


###############################################################################
# Inference functions:
###############################################################################

def assessNNSignificance(interactionDistribution,DF) :
    Beta = np.mean(interactionDistribution)
    Beta_SE = np.std(interactionDistribution)
    t_values = Beta / Beta_SE
    p = stats.t.sf(np.abs(t_values), DF)*2
    return(p)

###############################################################################
# Helper utils
###############################################################################

# Conv1D definition [in_channels, out_channels, kernel_size]  # in_channels = # SNPs, out_channels = number of neurons/filters
# Conv1D expected input shape [batch-size, in_channels, out_channels] # 
#def getConv1DOutput(myCov1D) : # get the shape, their product equals to the number of flat features
#    Cout = myCov1D.out_channels
#    Lout = int ( ( myCov1D.in_channels + 2*myCov1D.padding[0] - myCov1D.dilation[0]*(myCov1D.kernel_size[0] -1) -1 ) / myCov1D.stride[0] +1 ) # https://pytorch.org/docs/stable/nn.html#torch.nn.Conv1d
#    return( [Cout,Lout] )


def findPrevNextWeightedLayer(model,startIndex = -1, prev = True, startIndexOK = True): # finds the next/prev 'proper' layer with weights
    model = getModel(model)
    layersAsList = list(model)
    if startIndex == -1 : startIndex = len(layersAsList) -1 # need to find the actual layer index if we were passed in the python shortcut of -1 
    
    if prev : step = -1 # depending if we want the next or previous layer, the step will be +1 or -1 to forward or backwards
    else : step = 1

    currentIndex = startIndex 
    if startIndexOK == False : currentIndex += step # if we can use the start index, then start from there
    while True:
        if currentIndex >= len(layersAsList) or currentIndex < 0: 
            raise Exception("run out of layers!")
            break
        if type(model[currentIndex]) == nn.Conv1d or type(model[currentIndex]) == Conv1dSame or type(model[currentIndex]) == nn.Linear  or type(model[currentIndex]) == nn.Conv2d : 
           # print("found layer at: ", currentIndex)
            break
        currentIndex += step
        print("currentIndex, is:" , currentIndex, " (out of num layers:", len(layersAsList) ,")for type(model[currentIndex]):" , type(model[currentIndex]))

    return(currentIndex)
    

def isLayerActivation(layer) :
    if type(layer) == nn.Sigmoid or type(layer) == nn.ReLU or type(layer) == nn.LeakyReLU or type(layer) == nn.Softplus or type(layer) == nn.SELU : return(True)
    else : return(False)


def findPrevNextActivationLayer(model,layerIndex, prev = True, startIndexOK = True) : # finds the next activation type layer (RELU/leaky relu / SELu etc)
    if prev : step = -1 # depending if we want the next or previous layer, the step will be +1 or -1 to forward or backwards
    else : step = 1
    model = getModel(model)
    layersAsList = list(model)
    currentIndex = layerIndex 
    if startIndexOK == False : currentIndex += step # if we can use the start index, then start from there
    while True:
        if currentIndex >= len(layersAsList) or currentIndex < 0:  
            print("No Activation found!")
            currentIndex = -1
            break
        if (isLayerActivation(model[currentIndex])) : break
        #if type(model[currentIndex]) == nn.Sigmoid or type(model[currentIndex]) == nn.ReLU or type(model[currentIndex]) == nn.LeakyReLU or type(model[currentIndex]) == nn.Softplus or type(model[currentIndex]) == nn.SELU : 
           # print("found layer at: ", currentIndex)
        #    break 
        currentIndex += step
        
    return(currentIndex)


def setModelMode(model, training = True): # just setting model.training = False, is NOT enough to set all layers to be in 'eval' mode, it will only set the wrapper
    if training : model.train()
    else : model.eval()
 



        
def getModel(model) : # in case we need to access the underlying model of a DatParallelised model, dont know if this will return a model that has all the Weights correct or not...
    if type(model ) == nn.DataParallel : return model.module
    else: return model


def getFirstFCLayerIndex(model) : # in case we need to access the underlying model of a DatParallelised model, dont know if this will return a model that has all the Weights correct or not...
    model = getModel(model)
    if isinstance(model,nn.Sequential ) : return(0) # for sequential models the first usable layer where we can apply h2, is always the 0th index
    else : return (model.firstFCLayerIndex)  # if it not sequential then it must be a 'Tree model', then we have saved this into a variable
        

#    b_data_orig =train_X[0]
#    b_data = b_data_orig
#    b_data = OneHot(train_X[0]).astype('float32')
#    b_data.shape
   


    
def dataToDevice(b_data, device) :
    b_data =  torch.from_numpy(b_data ).to(device) 
    return(b_data)


    
