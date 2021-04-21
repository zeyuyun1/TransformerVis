import argparse
import numpy as np
import sys

# import sparsify
import sparsify_PyTorch

import torch
from tqdm import tqdm
import scipy as sp
import sklearn
from transformers import BertTokenizerFast, BertModel, BertForMaskedLM
from datasets import load_dataset

import random
import logging

from util import batch_up
from util import example_dim_old
from util import merge_two

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dictionary_dir', type=str, default = './dictionaries/dict_lcomplete_base_1500_epoch_1_d1500.npy',help=
                        'This is path for the a trained dictionary using train.py. The trained dictionary is a shape (hidden_state,dictionary_size) array saved as npy file.')
    
    parser.add_argument('--outfile_dir', type=str, default = './top_activate_examples/', help=
                        'The path where you want to save your output file.')
    
    parser.add_argument('--num_instances', type=int, default=298489, help='The number of sentences in our datasets. You can adjust this number to use a smaller datasets')
    
    parser.add_argument('--data_dir', type=str,default ='./data/sentences.npy', help=
                        'The path of data (a list of string, each string is a sentence of sequence of text with fixed length). The data is generated using data_generate.py. Since we dont need much data for dictionary learning (we can put all data in RAM at once), we save the data in npy file.')
    
    parser.add_argument('--gpu_id', type=int, default=0, help=
                        'The index that indicate which gpu you want to use')
    
    parser.add_argument('--l', type=int, default=1, help='Which layer of the transformer model we explain. For example, for BERT-base model, we can pick an layers from 0-12')
    
    parser.add_argument('--batch_size_1', type=int, default=10, help=
                        'This is the batch size for inference of transformer model. Basically, how many seqeuence we shove into our model at once. This number shouldnt be big because inference of transformer model took lots of memory.')
    
    parser.add_argument('--batch_size_2', type=int, default=100, help=
                        'This is the batch size for sparse code inference. This number can be big, but a batch size too big wouldnt really increase the speed of sparse enforce. Since its basically just an one layer neural network. Theres not much parrallel computing.')
    
    parser.add_argument('--num_transformer_factors', type=int, default=1000, help=
                        'The number of transformer factor we want to visualize. This must be smaller than the size of your dictionary')
    
    parser.add_argument('--shard_size', type=int, default=1000, help=
                        'TLDR: Make this number small if you have a memory error. This is number that indicates how much data (hidden states) that fits in your RAM at once. Recall that we are calculating the top-activated examples, so we need to calculate the top-n activations over the sparse code of all word vector. This is a really large number. Thus, we split this calculating max process in shards.')
    
    parser.add_argument('--reg', type=float, help=
                        'The regularization factor for sparse coding. You should use the same one you used in training')
    
    parser.add_argument('--top_n_activation', type=int, default=500, help=
                        'This number indicates how many examples do we collect for each transformer factor. By default, we collect top 200 activated examples.')
#     dict_lcomplete_base_2500_epoch_1_d2500_f.npy



    args = parser.parse_args()
    
#     laod model and tokenizer
    model_version = 'bert-base-uncased'
    tokenizer = BertTokenizerFast.from_pretrained(model_version)
    model = BertModel.from_pretrained(model_version)
    torch.cuda.set_device(args.gpu_id)
    device = torch.device("cuda:{}".format(args.gpu_id))
    model = model.to(device)
    
#     Load data
    sentences = np.load(args.data_dir).tolist()[:args.num_instances]
    
#     building dictionaries to collect examples for each num_transformer_factors. Notice that this "dictionary" is just the python dictionary, it's not the dictionary we used for dictionary learning.
    good_examples_contents = {}
    for d in range(args.num_transformer_factors):
        good_examples_contents[d] = []
#     good_examples_contents_new = {}
#     for d in range(args.num_transformer_factors):
#         good_examples_contents_new[d] = []

#     shard our data set into piece to fit into RAM
    sentences_shards = list(batch_up(sentences,batch_size=args.shard_size))
    logging.info("Numbers of sentences: {}".format(len(sentences)))
    
#     start the process to collect top activated example
    for shard_num in tqdm(range(len(sentences_shards)-1),'shards'):
#       define some parameters use for laters
        sentences_str = []
        words = []
        word_to_sentence = {} # word indice -> sentence (it belongs to) indice （sentence index, position in the sentence）
        sentence_to_word = {}
        n1=0
        n2=0
        X_set=[]
#       put our data into batch and ready to feed into transformer model
        sentences_batched = list(batch_up(sentences_shards[shard_num],batch_size=args.batch_size_1))
        for batch_idx in tqdm(range(len(sentences_batched)),'collect hidden states'):
            
#           This parts of the code looks complicated, but it basically keep track of a map between the word in each sentence 
#           to each of those sentences for convinience
            batch=sentences_batched[batch_idx]
            inputs_no_pad = tokenizer.batch_encode_plus(batch,add_special_tokens=False)
            inputs_no_pad_ids = inputs_no_pad['input_ids']
            len_ls = [ len(s) for s in inputs_no_pad_ids]
            inputs = tokenizer.batch_encode_plus(batch,return_tensors='pt', add_special_tokens=False,padding=True,truncation=True).to(device)
            for tokens in inputs_no_pad_ids:
                tokenized = tokenizer.convert_ids_to_tokens(tokens)
                sentences_str.append(tokenized)
                words.extend(tokenized)
                w_index = []
                for j in range(len(tokenized)):
                    word_to_sentence[n2] = (n1,j)
                    w_index.append(n2)
                    n2+=1
                sentence_to_word[n1] = w_index
                n1+=1
                
#           Collect hidden_states of a particular layers from the Transformer model. We also concadenate the hidden states of each 
#           sentences (a sequence of vectors) into a giant list (we use this later for sparse code inferences).
            hidden_states = model(**inputs,output_hidden_states=True)[-1]
            X=hidden_states[args.l].cpu().detach().numpy()
            for i in range(len(X)):
                sentences_trunc = X[i][:len_ls[i]]
                for s in range(len(sentences_trunc)):
                    X_set.append(sentences_trunc[s])

                
#       sparse code inference:

#       load dictionary
        basis1 = torch.from_numpy(np.load(args.dictionary_dir)).cuda()
#       we batch the hidden states we collected from the last steps using a larget batch size
        X_set_batched = list(batch_up(X_set,args.batch_size_2))
        X_sparse_set = []
        for i in tqdm(range(len(X_set_batched)),'sparse_inference'):
            batch = X_set_batched[i]
            I_cuda = torch.from_numpy(np.stack(batch, axis=1)).cuda()
            X_sparse = sparsify_PyTorch.FISTA(I_cuda, basis1, args.reg, 500)[0].T
            X_sparse_set.extend(X_sparse.cpu().detach().numpy())
            
#       We save the top n activated examples for each transformer factor in a dictionary. An examples contains the following: The word that corresponds to the embedding vector, the context sentence, the position of the word int he context sentence, the level of activation.
        for d in range(args.num_transformer_factors):
            good_examples_contents[d] = merge_two(example_dim_old(X_sparse_set,d,words,word_to_sentence,sentences_str,n=args.top_n_activation),good_examples_contents[d])[:args.top_n_activation]
            
#       save the examples, which are in python dictionaries
        np.save(args.outfile_dir + 'example_l_{}.npy'.format(args.l), good_examples_contents) 