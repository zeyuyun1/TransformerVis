This repo contains the code for paper: [Transformer visualization via dictionary learning: contextualized embedding as a linear superposition of transformer factors](https://arxiv.org/abs/2103.15949)

Instruction:

To visualize the hidden states for transformer factor. We need to first train a dictionary and then infer the sparse code using these dictionary.

To train a dictionary, 

```
run python train.py
```

If you want to use your own data, you need to put it in a python list, where each element is a string (sentences). Then save this list as a .npy file, then run

```
python train.py --training_data ./your_data.npy
```

To infer the sparse code and save the top activated examples for each transformer factors, run

```
python inference_efficient.py --dictionary_dir ./the_path_for_your_trained_dictionary
```
