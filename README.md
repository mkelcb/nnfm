![NNFM](https://github.com/mkelcb/nnfm/blob/master/nnfm_logo.png)

**NNFM: a neural-network fine-mapping tool**

This is a repository for the full source code of the NNFM project, which is a DNA sequence classifier CNN deep-learning model built on the Pytorch library.

##Objective
The model is motivated by the hypothesis that the number and arrangement of sequence motifs (eg. transcription factor binding sites) in the local region of GWAS associated SNPs may be used to predict the posterior probability of SNPs being causal to aid fine-mapping analyses. 

This code base is under active development, and I hope it will be eventually deployed as a hosted service.
In the meanwhile the code may be explored for educational purposes or run locally. A brief overview of each .py file follows:

**nnfmBinarizer:**
- helper functions to produce a binary from fasta sequences and converting to one-hot encoding

**nnfm_trainer:**
- the main NN model trainer

**nnfm_common:**
- common functionality used by both nnfm and nnfm_test including loading/standardising data and model construction

**nnfm:**
- main command line tool to be used for the training scenario

**nnfm_test:**
- main command line tool to be used for the test scenario


**Training outline:**
1. convert all chromosome fasta files into binaries 
2. prepare labels (see labels/21.label for an example)
3. Train model: a separate model should be trained for all 22 autosomes, so that a withheld estimate may be produced for any genomic coordinate

**Input:**
- fasta sequences, converted into binary representation
- optional auxilliary data 
- labels: chromosome, RSid, hg19 coordinates, A1,A2,... posterior probabilities for each trait


**Output:**
- Posterior probabilities of a each SNP being causal per trait

