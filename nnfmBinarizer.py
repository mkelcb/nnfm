# -*- coding: utf-8 -*-
"""
Created on Sun May  5 18:43:39 2019

@author: mk23
"""

# given a file with 1 sequence per line eg:

# CACTATTCTATCATGGTGCATTTTA
# GGTACCTATCGAGAACACCCTGCTG
# ...
# it transcodes them into a one-hot binary representation, and compresses the final results into .gzip
# optionally it also produces the reverse complements of the same, saved as _rev.sbin

import argparse
import numpy as np
import struct
#from shutil import copyfile
import os
#import gzip
#import sys
#from pathlib import Path

##################################################################
    
# If you only need to read the value of a global variable in a function, you don't need the global keyword: https://stackoverflow.com/questions/11867359/when-do-i-need-to-use-the-global-keyword-in-python
nucleotides="GCTA"
# the ordinal ascii codes of the lower case nucleotides
G=103
C=99
T=116
A=97
N=110

num_nucleotides=4 # there are 4 nucleotides, N is not a nucleotide, but its a 25% for all others

# the one hot codings  using the same alphabet as DEEPSEA
A_ = 0
G_ = 1
C_ = 2
T_ = 3

N_=4
N_value=0.25 # missing calls are represented as equal probability for all 4 nucleotides, IE 0.25
##################################################################

def getNumericOfNucleotide(base) : # convert  the nucleotides string representation to numeric
    if base == "G" or  base == "g" : return (G_)
    elif base == "C" or  base == "c" : return (C_)
    elif base == "T" or  base == "t" : return (T_)
    elif base == "A" or  base == "a" : return (A_)
    else : return(N_)

def get_d_code(dataType) :     # need to  convert dataType into single letter codes: https://docs.python.org/3/library/struct.html
    if( dataType == 'float32') : d_code = 'f'
    elif ( dataType == 'float16') : d_code = 'e' 
    elif ( dataType == 'int8') : d_code = 'b' 
    else : d_code = 'd' 
    return(d_code)
    

# location = "C:/softwares/Cluster/GIANT/miniPRS/dlpred/1_chrombin"
# loads an a single chromosome 
def loadChromFromDisk(location, dataType ="int8") :
    d_code = get_d_code(dataType)

    # open binary file, check if uncomrpessed file exist
    with open(location + ".sbin", "rb") as BinFile:
        BinFileContent = BinFile.read()

    # reformat data into correct dimensions
    sequence = np.array( struct.unpack(d_code*len(BinFileContent), BinFileContent  ), dtype = dataType )
    return(sequence)

    
def oneHot_to_numeric(seq_hot) :  # turns a a list of sequences in 3D one hot into a 2D array, IE: (n,nucleotides,p) ->  (n,p)
    numeric_seq= seq_hot.argmax(axis=1).astype(np.int8)
    
    # find indices for the Ns: want to find the indices where all elements are exactly 0.25       
    indices_N = np.all(seq_hot == N_value, axis=1)
    numeric_seq[indices_N] = N_
    return(numeric_seq )


def single_oneHot_to_numeric(seq_hot) :  # # turns a sequence in 2D one hot into a 1D array, IE: (nucleotides,p) ->  (p)
    numeric_seq = seq_hot.argmax(axis=0).astype(np.int8) 
    
    # find indices for the Ns: want to find the indices where all elements are exactly 0.25    
    indices_N = np.all(seq_hot == N_value, axis=0)
    numeric_seq[indices_N] = N_
    return(numeric_seq)
    
 
def revComplement(seq_num_rev) : # helper function that is used by both single and list versions
    origIndices_C = np.where(seq_num_rev == C_) # becase we overwrite them, we cannot do this in place
    origIndices_A = np.where(seq_num_rev == A_)
    seq_num_rev[np.where(seq_num_rev == G_)] = C_ # G -> C
    seq_num_rev[np.where(seq_num_rev == T_)] = A_ # T -> A
    seq_num_rev[origIndices_C] = G_ # C ->G
    seq_num_rev[origIndices_A] = T_ # A -> T 
    
    
def reverseComplement_numeric(sequence) : # converts a sequence in numeric format [1,2,0,3] to its reverse complement
    seq_num_rev = sequence.copy()
    seq_num_rev= seq_num_rev[:,::-1] # reversed sequence, this creates a VIEW of the original sequence, so changes applied to one will affect the other: https://stackoverflow.com/questions/6771428/most-efficient-way-to-reverse-a-numpy-array
    revComplement(seq_num_rev)
    return(seq_num_rev)
   
    
    #sequence = b_data
def single_reverseComplement(sequence) : # converts a sequence in numeric format [1,2,0,3] to its reverse complement
    seq_num_rev = sequence.copy()
    seq_num_rev= seq_num_rev[::-1] # reversed sequence, this creates a VIEW of the original sequence, so changes applied to one will affect the other: https://stackoverflow.com/questions/6771428/most-efficient-way-to-reverse-a-numpy-array
    revComplement(seq_num_rev)
    return(seq_num_rev)
     
# seq_onehot = b_data_hot
#seq_onehot[:,0] = np.array([0,0,1,0])
#seq_onehot[:,5] = np.array([0.3,0,0.7,0])

def singleOneHot_reverseComplement(seq_onehot) : # converts a sequence in Onhot format to its reverse complement
    # A_ = 0
    # G_ = 1
    # C_ = 2
    # T_ = 3
    # reverse complement: swap values according to:
    # C <-> G # swap values between C_ and G_ => [2, 1]
    # A <-> T # swap values between A_ and T_ => [0, 3]
    # N <-> N # Ns stay the same, but we can just apply the same logic and the 0.25s will stay the same
    seq_onehot_rev = seq_onehot.copy() # dont want to modify original
    seq_onehot_rev= seq_onehot_rev[:,::-1] # reversed sequence, this creates a VIEW of the original sequence, so changes applied to one will affect the other: https://stackoverflow.com/questions/6771428/most-efficient-way-to-reverse-a-numpy-array
    # WARNING, these swaps are alphabet specific!!
    swaps = [T_,C_,G_,A_] # rearrange indices, what was originally C will become G, and what was originally G becomes C, and so on
    seq_onehot_rev = seq_onehot_rev[swaps,:]

    return(seq_onehot_rev)



def OneHot(sequence,num_nucleotides =4) : # converts 2d numeric sequences of numeric representation of [0,1,2,3] into a 3D array of 1 hot // (n,128) -> (n,4,128)
    seq_onehot = np.zeros((num_nucleotides,sequence.shape[0],sequence.shape[1]), dtype=np.float32)
    for i in range(num_nucleotides):
        b = sequence == i # get a mask for the nucleotide, IE all places where it is 'G', this is n,128,
        seq_onehot[i, b] = 1    # set 1 to nucleotide    

    # add in the Ns as 0.25, where we had Ns in the original sequence
    indices_N = np.where(sequence == N_) # this gets a tuple, where [0] is the sequence number, and [1] are the nucleotides that are Ns
    seq_onehot[:,indices_N[0],indices_N[1]] = N_value        
    
    # this produces a shape (4, 271,128 ) : (nucleotides, n, p)
    # but we want (271,4, 128, ) : (n,nucleotides, p, )
    seq_onehot =  seq_onehot.transpose(1, 0, 2)
    return(seq_onehot)     


def single_OneHot(sequence,num_nucleotides =4) : # convert a single numeric representation of [0,1,2,3] into a 2D array of 1 hot
    seq_onehot = np.zeros((num_nucleotides,len(sequence)), dtype=np.float32)
    for i in range(num_nucleotides):
        b = sequence == i
        seq_onehot[i, b] = 1  

    # add in the Ns as 0.25, where we had Ns in the original sequence
    indices_N = np.where(sequence == N_)
    seq_onehot[:,indices_N] = N_value       
    
    return(seq_onehot)     
    
    



def oneHot_to_text(seq_hot) :  # converts a list of one hot sequences in 3D format, to one-hot representation of a sequence context back into text
    num_representation =  oneHot_to_numeric(seq_hot) 
    return(numeric_to_text(num_representation))

    
def numToText(num_representation) : # inner function used by both single and list numeric_to_text
    num_representation[np.where(num_representation == G_)] = G
    num_representation[np.where(num_representation == C_)] = C
    num_representation[np.where(num_representation == T_)] = T
    num_representation[np.where(num_representation == A_)] = A
    num_representation[np.where(num_representation == N_)] = N
    
    
def numeric_to_text(num_representation) :  # 2D numeric format to a list of strings, IE (n,128) -> list('GCTA')
    num_representation = num_representation.copy() # otherwise the line below would change the original
    # thranscode it back into chr codes
    numToText(num_representation)
    
    allSequences = list()
    for i in range(num_representation.shape[0]) :
        num_representation_i = num_representation[i]
        num_representation_i = num_representation_i.tolist()
        text_representation = [chr(num).upper() for num in num_representation_i] # obtain array of text
        text_representation = ''.join(text_representation) # convert array to string
        
        allSequences.append(text_representation)
    return(allSequences)
    
    
def single_numeric_to_text(num_representation) :   # converts a single numeric format to a list of strings, IE (128,) -> 'GCTA'
    num_representation = num_representation.copy() # otherwise the line below would change the original
    # thranscode it back into chr codes
    numToText(num_representation)
    
    num_representation = num_representation.tolist() 
    text_representation = [chr(num).upper() for num in num_representation] # obtain array of text
    text_representation = ''.join(text_representation) # convert array to string
    
    return(text_representation)
    


def single_text_to_numeric(seq_text) : # turns  single sequence text into numeric representation: 'GCTA' -> [0,1,2,3]
    seq_num = np.array([ord(char) for char in seq_text.lower()])  # https://stackoverflow.com/questions/4528982/convert-alphabet-letters-to-number-in-python
    # map each character to the nucleotide code (this works in one go, as none of the ordinal character codes map directly to the 0,1,2,3)
    seq_num[np.where(seq_num == G)] = G_
    seq_num[np.where(seq_num == C)] = C_
    seq_num[np.where(seq_num == T)] = T_
    seq_num[np.where(seq_num == A)] = A_
    seq_num[np.where(seq_num == N)] = N_ 
    return(seq_num)


def write_text_sequences(out,allSequences) :   # writes sequences which are in text format 'GCTAGCTA' to a text file
    with open(out +".seq", "w") as resultsFile: 
        #resultsFile.write("predictor1" + "\t" + "predictor2"  + "\n" )
        for i in range(len(allSequences)  ) :
            resultsFile.write(str(allSequences[i]) + "\n" )

  



#############################################################

# sequence/binary conversions for a SINGLE sequence, IE 2D not 3D


            
  
    
    
def single_text_to_OneHot(seq_text) : # wrapper function that performs both text to numeric, and numeric to One hot
    seq_num = single_text_to_numeric(seq_text)
    return( single_OneHot(seq_num) )
    
    


def single_oneHot_to_text(seq_hot) :  # converts a single 2D 1 hot representation of a sequence context back into text
    num_representation =  single_oneHot_to_numeric(seq_hot) 
    return(single_numeric_to_text(num_representation))
    
 #  num_representation =  sequence[0]

    

    

################################################################
    

def processChromosomeToBinary(seq,out) :

    #seq_text='CCTGGGCTGCCCCACCCCATGCCCAGCATGAGCCTGGAAGGGCCCCACCACACACCTTCTTGCGAGTGGTCATGGCGTTGCGCAGGTGTATGGCGAGCTGGCGGATGTAGAGGAAGGCGTGCTGGTAG'
    #seq_text='GCTAGCTA'

    with open(seq, "r") as seqFile:
        seq_text = seqFile.readline().rstrip()


    # 2) transcode it into numerical representation, that is [G,C,T,A,N] - > [0,1,2,3,4]
    seq_num = single_text_to_numeric(seq_text)
    
    flat = seq_num.ravel() # flatten the representation to numeric before 
    del seq_num # free up RAM      
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    # depending on if we wanted to compress the data, 

    writeSBinToDisk(flat, out)
    del flat
    
    # copy sequence ID to target location too
    print("written sequence binary to:" , out  +".sbin" )
    




    
def writeSBinToDisk(flat, outFile) :
    d_code = get_d_code("int8")
    # write binary to disk
    totalFlatLen= len(flat) 
    numParts = 10
    tenP = totalFlatLen//numParts # split it into 10 parts
    startIndex = 0
    for i in range(numParts) :  
        mode="ab"
        endIndex = startIndex + tenP
        if i == (numParts-1) : endIndex = totalFlatLen # the last part should cover up until the end
        if i == 0: mode = "wb" # for the first time we want it to start a new file
        
        print("startIndex:" , startIndex, "endIndex is:" , endIndex, " out of totalFlatLen: ", totalFlatLen)
        flatData = struct.pack(d_code*len(flat[startIndex:endIndex]),*flat[startIndex:endIndex]  )
        with open(outFile + ".sbin", mode ) as flat_File: 
            flat_File.write(flatData) 
        
        startIndex = endIndex


#onebased = True
#start=126
#end=131
#padding = True
#if onebased: start -=1 # if coordinates were supplied in one based format, need to offset start by -1
#else : end += 1
#end - start

def extractWindowFromSequence(sequence,start,end,onebased = True, padding = False)  :
    if onebased: start -=1 # if coordinates were supplied in one based format, need to offset start by -1
    else : end += 1 # if it was the other way around we must go +1 at the end
    window_size = end - start
    
    startPadding = 0
    if start < 0 : 
        startPadding = abs(0 - start)
        start = 0  
        
    endPadding = 0
    if end > len(sequence) : 
        endPadding = abs(end - len(sequence) )
        end = len(sequence)

    # extract sequence
    window_numeric = sequence[start:end]
    
    # if padding was requested, and the sequence extracted is not as long as it should be
    if padding and window_size != len(window_numeric):
        if startPadding > 0 : # if too short at the start
            pad = np.ones( startPadding ,dtype=np.int8) * N_
            window_numeric = np.concatenate( (pad,window_numeric) )


        if endPadding > 0 : # if too short at the end
            pad = np.ones( endPadding ,dtype=np.int8) * N_
            window_numeric = np.concatenate( (window_numeric, pad) )

    return(window_numeric)
        
    
# 1) load all auxilliary data for all SNPs
# CHR,BP,SNP,A1,A2,MAF
def loadAuxData(auxdata, keyBySNP = True) : # loads all labels from the used blocks
    auxData = {}
    counter = 0
    with open(auxdata, "r") as id:
        for i in id:
            counter+=1
            if counter == 1 : continue 
            itmp = i.rstrip().split()
            
            CHR = int(itmp[0])
            BP = int(itmp[1])
            SNP = itmp[2]
            A1 = getNumericOfNucleotide(itmp[3]) # transcode A1/A2 into numeric representations
            A2 = getNumericOfNucleotide(itmp[4])
            MAF = float(itmp[5])
            
            # 2 options to key the dict, by SNP id or by bp
            if keyBySNP : auxData[SNP] = [CHR,BP,SNP,A1,A2,MAF]
            else : auxData[BP] = [CHR,BP,SNP,A1,A2,MAF]

    return(auxData)

        
##################################################

#out="C:/softwares/Cluster/GIANT/miniPRS/dlpred/1_chrombin"
#seq="C:/softwares/Cluster/GIANT/miniPRS/dlpred/1"
#reverseCompliment= True
def runMain(args) :
    args.onebased = args.onebased == 1
    print('Loading chromosome data from ',  args.seq)     
    
    if args.start > -1 :
        print('extracting location from', args.start, "to:",args.end, "(onebased: ",args.onebased,")")        
        sequence = loadChromFromDisk(args.seq)
        window_numeric = extractWindowFromSequence(sequence,args.start,args.end,args.onebased)
        window_text = single_numeric_to_text(window_numeric)
        write_text_sequences(args.out,[window_text])
        print("written extracted window to",args.out )
    else :
        print('converting to binary')   
        processChromosomeToBinary(args.seq,args.out)
    
    
    
if __name__ == '__main__':   
    parser = argparse.ArgumentParser()

    parser.add_argument("--out",required=True, help='an output location is always required')
    parser.add_argument("--seq",required=True, help='The name of the chromosome file in text format')  

    parser.add_argument("--start",required=False, help='the start location of the extraction (-1 to disable)', default=-1, type=int)  
    parser.add_argument("--end",required=False, help='the end location of the extraction', default=-1, type=int)  

    parser.add_argument("--onebased",required=False, help='if the coordinates of the location is expected in 1 based (1) or zero based(0) positions (default 1)', default=1, type=int)  

    parser.set_defaults(func=runMain)
        
    args = parser.parse_args()
    args.func(args)


    # consider gzip (IE if the 3D matrix is 4 times larger, but of mostly zeros, then gzipping it may be a good idea)
