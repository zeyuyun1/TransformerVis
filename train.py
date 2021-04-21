import argparse
# import imageio
import numpy as np
import numpy.linalg as la
import scipy.io
import sys

# import sparsify
import sparsify_PyTorch

import torch
from tqdm import tqdm
import scipy as sp
import sklearn
from transformers import BertTokenizerFast, BertModel, BertForMaskedLM
from datasets import load_dataset
import nltk
from nltk.probability import FreqDist
from sklearn.datasets import load_digits
import logging

def batch_up(iterable, batch_size=1):
    l = len(iterable)
    for ndx in range(0, l, batch_size):
        yield iterable[ndx:min(ndx + batch_size, l)]


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=0, help=
                        'The index that indicate which gpu you want to use')
    
    parser.add_argument('--num_instances', type=int, default=298489, help=
                        'The number of sentences in our datasets. You can adjust this number to use a smaller datasets')
    
    parser.add_argument('--epoches', type=int, default=2, help=
                        'numbers of epoch you want to train your dictionary')
    
    parser.add_argument('--PHI_NUM', type=int, default=2000, help=
                        'The size of the dictionary. Also equivalent to the number of transformer factors.')
    
    parser.add_argument('--HIDDEN_DIM', type=int, default=768, help=
                        'The size of hidden state of your transformer model. This depends on the model you use. The default the size of hidden states of BERT base')

    parser.add_argument('--batch_size_1', type=int, default=10, help=
                        'This is the batch size for inference of transformer model. Basically, how many seqeuence we shove into our model at once. This number shouldnt be big because inference of transformer model took lots of memory.')
    
    parser.add_argument('--batch_size_2', type=int, default=100, help=
                        'This is the batch size for sparse code inference. This number can be big, but a batch size too big wouldnt really increase the speed of sparse enforce. Since its basically just an one layer neural network. Theres not much parrallel computing.')
    
    parser.add_argument('--reg', type=float, default=0.3, help=
                        'The regularization factor for sparse coding. You should use the same one you used in inference ')
    
    parser.add_argument('--load', type=str, default=None, help=
                        'Instead of intialize an random dictionary for training. You can also enter a path here indicating the the path of the dictionary you want to start with. The file must be a .npy file')
    
    parser.add_argument('--training_data', type=str, default='./data/sentences_short.npy', help=
                        'path of training data file. Again, must be a .npy file')
    
    parser.add_argument('--name', type=str, default='short', 
                        help='The name you want to have for your trained dictionary file  ')

    parser.add_argument('--model_version', type=str, default='bert-base-uncased', help='The model you want to use for your transformer model. The current code only support bert-base-uncased and bert-large-uncased')    
    
    args = parser.parse_args()
    filename_save = '''./dictionaries/{}_{}_reg{}_d{}_epoch{}'''.format(args.model_version,args.name,args.reg,args.PHI_NUM,args.epoches)
    
#     laod model and tokenizer
    model_version = args.model_version
    tokenizer = BertTokenizerFast.from_pretrained(model_version)
    model = BertModel.from_pretrained(model_version)
    torch.cuda.set_device(args.gpu_id)
    device = torch.device("cuda:{}".format(args.gpu_id))
    model = model.to(device)
    
#   load data
    sentences = np.load(args.training_data).tolist()[:args.num_instances]
    
    print("Numbers of sentences: {}".format(len(sentences)))
#   collect the frequency of each word in our training data. The word with high freqeuncy should receive a smaller weight
#   during the dictionary update. We took care of this in our training loop. The reason for doing this is explained in       the appendix 
    words = []
    for s in sentences:
        words.extend(tokenizer.tokenize(s))
    data_analysis = nltk.FreqDist(words)
    for w in data_analysis:
        data_analysis[w] = np.sqrt(data_analysis[w])
        
#   initilize the dictionary matrix
    PHI_SIZE = [args.HIDDEN_DIM, args.PHI_NUM]
    PHI = torch.randn(PHI_SIZE).to(device)
    PHI = PHI.div_(PHI.norm(2,0))
    
#   or you can also load a dictionary. You might want to do this if you are high way trough training a dictionary. And you want to keep training it.
    if args.load:
        print('load from: '+ args.load)
        PHI = torch.from_numpy(np.load(args.load)).to(device)

#   intialize some variable used in dictionary learning
    lambd = 1.0
    ACT_HISTORY_LEN = 300
    HessianDiag = torch.zeros(args.PHI_NUM).to(device)
    ActL1 = torch.zeros(args.PHI_NUM).to(device)
    signalEnergy = 0.
    noiseEnergy = 0.
    X_att_set_temp = []
    frequency_temp = []
    
#   This is the layers in transformer model that we collect hidden states. In the paper, we suggest to collect hidden states from every layers, i.e. 0,1,2,...,11. In the actual implementation, we collect hidden states from every other layers to reduce computation.
    if args.model_version == 'bert-base-uncased':
        layers= [0,2,4,6,8,10]
    else:
        layers = [0,2,4,6,8,10,12,14,16,18,20,22]

#     I = np.zeros([args.HIDDEN_DIM,args.batch_size_2]).astype('float32')

#   starting the dictionary training loop, the training loop is divided into the following 2 steps:
#   1. collect hidden states from transformer. Once we collect enough those hidden state vector, we jump to step 2.
#   2. Use the hidden state vectors collect from step 1 to update the dictionary. Once we are done with exhuast those hidden states. We jump back step 1 to collect more of those hidden states.
    sentences_batched = list(batch_up(sentences,batch_size=args.batch_size_1))
    for epoch in range(args.epoches):
        print("Epoch: {}".format(epoch))
        
#       Step 1: collecting hidden states using different input sentences from transformer model: 
        for batch_idx in tqdm(range(len(sentences_batched)),'main loop'):
            if batch_idx%100==0:
                #save your dictionary every now and then to avoid the unexpected crash during training loop:
                np.save(filename_save, PHI.cpu().detach().numpy())
            batch = sentences_batched[batch_idx]
            inputs_no_pad = tokenizer.batch_encode_plus(batch,add_special_tokens=False)
            inputs_no_pad_ids = inputs_no_pad['input_ids']
            len_ls = [ len(s) for s in inputs_no_pad_ids]
            inputs = tokenizer.batch_encode_plus(batch,return_tensors='pt', add_special_tokens=False,padding=True,truncation=True).to(device)

            hidden_states = model(**inputs,output_hidden_states=True)[-1]
            for l in layers:
                X=hidden_states[l].cpu().detach().numpy()

                for i in range(len(X)):
                    sentences_trunc = X[i][:len_ls[i]]
                    for s in range(len(sentences_trunc)):
                        X_set_temp.append(sentences_trunc[s])

            # update word/sentence tracker and frequency
                for tokens in inputs_no_pad_ids:
                    tokenized = tokenizer.convert_ids_to_tokens(tokens)
                    frequency_temp.extend([data_analysis[w] if w in data_analysis else 1 for w in tokenized])


        #   Step 2: once we collece enough hidden states, we train the dictionary.
            if batch_idx%5==0 and batch_idx>0:
                X_set_batched = list(batch_up(X_set_temp,args.batch_size_2))
                words_frequency_batched = list(batch_up(frequency_temp,args.batch_size_2))
        #         print('start_training_dictionary:')

                for i in tqdm(range(len(X_set_batched)),'update dictionary'):
                    batch = X_set_batched[i]
                    I_cuda = torch.from_numpy(np.stack(batch, axis=1)).to(device)
                    frequency = torch.tensor(words_frequency_batched[i]).float().to(device)
                    ahat, Res = sparsify_PyTorch.FISTA(I_cuda, PHI, args.reg, 500)

                    #Statistics Collection
                    ActL1 = ActL1.mul((ACT_HISTORY_LEN-1.0)/ACT_HISTORY_LEN) + ahat.abs().mean(1)/ACT_HISTORY_LEN
                    HessianDiag = HessianDiag.mul((ACT_HISTORY_LEN-1.0)/ACT_HISTORY_LEN) + torch.pow(ahat,2).mean(1)/ACT_HISTORY_LEN

                    signalEnergy = signalEnergy*((ACT_HISTORY_LEN-1.0)/ACT_HISTORY_LEN) + torch.pow(I_cuda,2).sum()/ACT_HISTORY_LEN
                    noiseEnergy = noiseEnergy*((ACT_HISTORY_LEN-1.0)/ACT_HISTORY_LEN) + torch.pow(Res,2).sum()/ACT_HISTORY_LEN
                    snr = signalEnergy/noiseEnergy

                    #Dictionary Update
                    PHI = sparsify_PyTorch.quadraticBasisUpdate(PHI, Res*(1/frequency), ahat, 0.001, HessianDiag, 0.005)

#               At this points, we finish exhuast all the hidden states we collect to update the dictionary. So we will dump all the hidden states vectors and jump back to step 1. We also print our some statistic for dictionary training so one can check how good their training are.
                print("Total_step {a}, snr: {b}, act1 max: {c}, act1 min: {d}".format(a=epoch, b=snr,c = ActL1.max(),d=ActL1.min()))
                X_set_temp=[]
                frequency_temp=[]
    
    np.save(filename_save, PHI.cpu().detach().numpy())
    