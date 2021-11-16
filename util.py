import numpy as np
import sys
import torch
from tqdm import tqdm
import scipy as sp
import sklearn
import torch.nn.functional as F
from IPython.display import HTML as html_print
from matplotlib import colors
import string
from math import log, e

# import sparsify
import sparsify_PyTorch

# import lime
from lime.lime_text import LimeTextExplainer
from transformers import BertTokenizerFast

result = string.punctuation
model_version = 'bert-base-uncased'
tokenizer = BertTokenizerFast.from_pretrained(model_version)
# model = BertModel.from_pretrained(model_version)

def merge_two(list1, list2):
    ls = list1 + list2
    ls.sort(key = lambda x: x['score'],reverse=True)
    return ls

def batch_up(iterable, batch_size=1):
    l = len(iterable)
    for ndx in range(0, l, batch_size):
        yield iterable[ndx:min(ndx + batch_size, l)]

        
def example_dim_old(source, dim ,words,word_to_sentence,sentences_str,n=5,head=None,verbose = True,vis=False):
    my_df = []
    dim_slices = [x[dim] for x in source]
    indx = np.argsort(-np.array(dim_slices))[:n]
#     indx = np.argpartition(dim_slices,-n)[-n:]
    for i in indx:
        word = words[i]
        act = dim_slices[i]
        sent_position, word_position = word_to_sentence[i]
        sentence= sentences_str[sent_position]
        d = {'word':word,'index':word_position,'sent_index':sent_position,'score':act,'sent':tokenizer.decode(tokenizer.convert_tokens_to_ids(sentence)),}
        my_df.append(d)
    return my_df

def cstr(s, color='black'):
    return "<text style=color:{}>{}</text>".format(color, s)

def blend(alpha, color =[255,0,0], base=[255,255,255]):
    '''
    color should be a 3-element iterable,  elements in [0,255]
    alpha should be a float in [0,1]
    base should be a 3-element iterable, elements in [0,255] (defaults to white)
    '''
    out = [int(round((alpha * color[i]) + ((1 - alpha) * base[i])))/255 for i in range(3)]
    return colors.to_hex(out)
def cstr_color(s, opacity=0, switch= 0):
    if switch:
        return "<text style=background-color:{};>{}</text>".format(blend(opacity,color =[255,50,0]),s)
    else:
        return "<text style=background-color:{};>{}</text>".format(blend(opacity,color =[50,255,0]),s)

        
def decode_color(token,salient_map,word_index):
#     dict ids
    temp = 0
    if word_index in salient_map:
        temp = salient_map[word_index]
    salient_map[word_index]=0
    max_weight = max(0.01,max(salient_map.values()))
    min_weight = min(-0.01,min(salient_map.values()))
    if temp<=max_weight:
        salient_map[word_index] = temp
    if temp>max_weight:
        salient_map[word_index] = max_weight

    max_weight = max(0.01,max(salient_map.values()))
    min_weight = min(-0.01,min(salient_map.values()))
    
    sent = ''
    for i in range(len(token)):
        w = token[i]
        if token[i][0]=='#':
            w = w[2:]
        if i==word_index:
            w= "<text style=color:{}>{}</text>".format('blue', w)
        if i in salient_map:
            if salient_map[i]>0:
                switch = 1
                opacity = salient_map[i]/max_weight
            else:
                switch = 0
                opacity = abs(salient_map[i])/abs(min_weight)
            w = cstr_color(w,opacity = opacity,switch = switch)
        if token[i][0]=='#' or token[i][0] in string.punctuation:
            sent +=w
        else:
            sent = sent +' '+ w
    return sent

def decode(token,ids):
    sent = ''
    for i in range(len(token)):
        w = token[i]
        if token[i][0]=='#':
            w = w[2:]
        if i in ids:
            w = cstr(w,color ='blue')
        if token[i][0]=='#' or token[i][0] in string.punctuation:
            sent +=w
        else:
            sent = sent +' '+ w
    return sent 

def generate_salient_map(model,l,basis1,text_instance,word_index,sparse_dim,num_features,num_samples,BATCH_SIZE_1,BATCH_SIZE_2,reg,feature_selection='auto'):
#     this function is modified from the LimeTextExplainer function from the lime repo:
#     https://github.com/marcotcr/lime/blob/a2c7a6fb70bce2e089cb146a31f483bf218875eb/lime/lime_text.py#L301
#     I modified it to fit the huggingface style bert tokenizer.

    model_regressor=None
    explainer = LimeTextExplainer()
    encode = tokenizer(text_instance, add_special_tokens=False)['input_ids']

    inputs = [encode for i in range(num_samples)]

    distance_metric='cosine'


    def distance_fn(x):
        return sklearn.metrics.pairwise.pairwise_distances(
            x, x[0], metric=distance_metric).ravel() * 100
    def classifier_fn(inputs):

#         hook_1 = Save_int()
#         handle_1 = model.encoder.layer[l-1].attention.output.dropout.register_forward_hook(hook_1)
        #     inputs = tokenizer(str_to_predict,return_tensors='pt', add_special_tokens=False).cuda()
        I_cuda_ls = []
        inputs_batched = batch_up(inputs,BATCH_SIZE_1)
        for inputs in inputs_batched:
            inputs = torch.tensor(inputs).cuda()
            hidden_states = model(inputs,output_hidden_states=True)[-1]
            X_att=hidden_states[l].cpu().detach().numpy()

            I_cuda_ls.extend(X_att[:,word_index,:])
        result= []
        I_cuda_batched = batch_up(I_cuda_ls,BATCH_SIZE_2)
        for batch in I_cuda_batched:
            I_cuda = torch.from_numpy(np.stack(batch, axis=1)).cuda()
            X_att_sparse = sparsify_PyTorch.FISTA(I_cuda, basis1, reg, 1000)[0].T
            result.extend(X_att_sparse[:,sparse_dim].cpu().detach().numpy())
    #     print(np.array(result).shape)

        return np.array(result).reshape(-1,1)

    doc_size = len(encode)
    sample = np.random.randint(1, doc_size + 1, num_samples - 1)
    data = np.ones((num_samples, doc_size))
    data[0] = np.ones(doc_size)
    features_range = range(doc_size)
    for i, size in enumerate(sample, start=1):
        inactive = np.random.choice(features_range, size,
                                            replace=False)
        data[i, inactive] = 0

    inverse_data= np.array(inputs)
    inverse_data[~data.astype('bool')]=tokenizer.convert_tokens_to_ids(tokenizer.unk_token)
    distances = distance_fn(sp.sparse.csr_matrix(data))
    yss = classifier_fn(inverse_data)

    salient_map =dict(explainer.base.explain_instance_with_data(
                    data, yss, distances, 0, num_features,
                    model_regressor=model_regressor,
                    feature_selection=feature_selection)[1])
    
    return salient_map
    
def print_example_with_saliency(model,l,basis1,examples,sparse_dim,num_features =10,num_samples = 1051,repeat = False,BATCH_SIZE_1=8,BATCH_SIZE_2=200,reg=0.3,feature_selection='auto'):
    # text_instance = """music in the uk and ireland stating that the single" welds a killer falsetto chorus to a latterday incarnation of the' wall of sound'"."""
    final_print=''
    all_sentences={}
    for example in tqdm(examples):
        word_index = example['index']
        sent_index = example['sent_index']
        text_instance = example['sent']

        tokens = tokenizer.tokenize(text_instance)
#         if len(tokens)>70:
#             continue
        salient_map = generate_salient_map(model,l,basis1,text_instance,word_index,sparse_dim,num_features,num_samples,BATCH_SIZE_1,BATCH_SIZE_2,reg,feature_selection = feature_selection)
        result = decode_color(tokens,salient_map,word_index)
        if sent_index not in all_sentences:
            all_sentences[sent_index] = [result]
        else:
            if repeat:
                all_sentences[sent_index].append(result)
        #if we don't want repeated sentences
            else:
                continue
        if len(all_sentences)>20:
            break
    for ls in all_sentences.values():
        for block in ls:
            final_print = final_print + block + '<br />' + '<br />'
    return final_print

def print_example(examples,n=70):
    
    sentence_to_print = {}
    for example in examples[:n]:
        word = example['word']
        act = example['score']
        sent_position = example['sent_index']
        word_position = example['index']
        sents = example['sent']

        if sent_position not in sentence_to_print:
            sentence= tokenizer.tokenize(sents)
            tokens = sentence.copy()
            sentence_to_print[sent_position] = tokens,[],[]
        sentence_to_print[sent_position][1].append(word_position)
        sentence_to_print[sent_position][2].append(act)
#         sentence_to_print[sent_position][word_position] = cstr(sentence_to_print[sent_position][word_position], color='red')
    final_print=''
    
    values = list(sentence_to_print.values())
    
    for i in range(len(values)-1):
        act = values[i][-1]
        token,ids = values[i+1][:-1]
        result =  decode(token,ids)
        if i==0:
            result = ''
        final_print =  final_print + result + '<br />' + '<br />' + 'Sparse Coding Activation: {}  '.format([act[0]]) +'Sentence:  '
#     html_print(cstr(final_print, color='black'))
    return final_print


def print_example_simple(examples,window=10,n=70):
    final_print='0'
    for i,example in enumerate(examples[:n]):
        word_position = example['index']
        sents = example['sent']
        sentence= tokenizer.tokenize(sents)
        tokens = sentence.copy()[max(0,word_position-window):min(word_position+window,len(sentence))]
        word_position = min(window,word_position)
        result =  decode(tokens,[word_position])
        final_print =  final_print + result + '<br />'+str(i+1)
    return final_print


def entropy(labels, base=None):
    n_labels = len(labels)

    if n_labels <= 1:
        return 0
    value,counts = np.unique(labels, return_counts=True)
    probs = counts / n_labels
    n_classes = np.count_nonzero(probs)

    if n_classes <= 1:
        return 0

    ent = 0.

    # Compute entropy
    base = e if base is None else base
    for i in probs:
        ent -= i * log(i, base)

    return ent

