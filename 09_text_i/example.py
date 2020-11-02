#!/usr/bin/python3

# these lines prevent lots of warnings from being displayed
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# import the data mining libraries
import torch
import transformers

# load the model
# NOTE:
# see https://huggingface.co/transformers/pretrained_models.html for more examples
#feature_generator = 'bert-base-uncased'
#feature_generator = 'bert-base-multilingual-uncased'
feature_generator = 'albert-base-v2'

tokenizer = transformers.AutoTokenizer.from_pretrained(feature_generator)
feature_generator = transformers.AutoModel.from_pretrained(feature_generator)

def make_features(x):
    encoding = tokenizer.encode_plus(
        x,
        max_length = 64,
        truncation = True,
        pad_to_max_length = True,
        return_tensors = 'pt',
        )
    with torch.no_grad():
        last_layer,embedding = feature_generator(**encoding) 
    features = torch.mean(last_layer,dim=1)
    return features

features = make_features('this is a test')
print("features.shape=",features.shape)



