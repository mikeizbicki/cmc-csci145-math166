#!/usr/bin/python3

# process command line arguments
import argparse
parser = argparse.ArgumentParser()

parser_options = parser.add_argument_group('options')
parser_options.add_argument('--mode',choices=['feature_generation','train'],default='train')
parser_options.add_argument('--datapath',required=True)
parser_options.add_argument('--class_labels',default='class_labels.json')

parser_hyperparam = parser.add_argument_group('hyperparameters')
parser_hyperparam.add_argument('--eta', type=float, required=True)
parser_hyperparam.add_argument('--e', type=int, default=2)
parser_hyperparam.add_argument('--lambda', dest='lambda_', type=float, default=0.0)
parser_hyperparam.add_argument('--optimizer',default='SGD')
parser_hyperparam.add_argument('--batch_size',type=int,default=32)
parser_hyperparam.add_argument('--feature_generator',default='albert-base-v2')

parser_debug = parser.add_argument_group('debug')
parser_debug.add_argument('--logdir', default='log')
parser_debug.add_argument('--seed', default=0)
parser_debug.add_argument('--device',choices=['auto','cpu','gpu'],default='auto')
args = parser.parse_args()

# load libraries
import os
import random
random.seed(args.seed)
import sys
import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    )

# NOTE:
# simplejson is a faster version of the json library that is built into python;
# you can install it with `pip3 install simplejson`,
# or you can change the line to `import json`
import simplejson as json

# load tensorboard
# NOTE:
# tensorboard is part of the tensorflow library, not pytorch;
# tensorflow has an annoying habbit of printing tons of debugging information;
# the os.environ line disables printing this information before performing the import
# see: https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from torch.utils.tensorboard import SummaryWriter

# prevent lots of warnings from being displayed
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import torch
torch.manual_seed(args.seed)

# import the data mining libraries
import torch
import transformers

# set device to cpu/gpu
if args.device=='gpu' or (torch.cuda.is_available() and args.device=='auto'):
    device = torch.device('cuda')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    device = torch.device('cpu')
logging.info('device='+str(device))

# generate directory for tensorboard logs
if args.mode=='train':
    logdir = os.path.join(
        args.logdir,
        'feature_generator='+args.feature_generator+',optimizer='+args.optimizer+',eta='+str(args.eta)+'batch_size='+str(args.batch_size)+',e='+str(args.e)+',lambda='+str(args.lambda_)
        )
    try:
        os.makedirs(logdir)
        writer = SummaryWriter(logdir)
    except FileExistsError:
        logging.error('logdir='+logdir+' already exists')
        import sys
        sys.exit(1)

# load the class labels
# NOTE:
# the class_labels variable is a list of all the screen_name values in the dataset;
# the index in the list is the number representation of the class;
# the args.class_labels file should be created using make_class_labels.py
with open(args.class_labels) as f:
    class_labels = sorted(json.loads(f.read()).keys())

# load the dataset
from torch.utils.data import DataLoader, IterableDataset

class Dataset(IterableDataset):
    '''
    this loads the dataset using O(1) memory instead of O(n) memory
    '''
    def __iter__(self):
        return Dataset.scramble(Dataset.yield_data(), 8096)

    def yield_data():
        '''
        creates a generator that yields a single tweet at a time;
        the results are yielded in the order they are present on disk,
        and so the results must be combined with the scramble function 
        to generate a random sample
        '''
        filenames = sorted(os.listdir(args.datapath))
        random.shuffle(filenames)
        for filename in filenames:
            filepath = os.path.join(args.datapath,filename)
            with open(filepath) as f:
                for line in f:
                    tweets = json.loads(line)
                    if type(tweets) is list:
                        for tweet in tweets:
                            tweet['label_id'] = class_labels.index(tweet['screen_name'])
                            yield tweet
                    else:
                        tweet = tweets
                        tweet['features'] = torch.tensor(tweet['features'])
                        tweet['label_id'] = class_labels.index(tweet['screen_name'])
                        yield tweet

    def scramble(gen, buffer_size):
        '''
        randomizes the order of a generator using O(buffer_size) memory instead of O(n);
        the downside is that the results are not truly uniformly random,
        but they are random enough for training purposes
        '''
        buf = []
        i = iter(gen)
        while True:
            try:
                e = next(i)
                buf.append(e)
                if len(buf) >= buffer_size:
                    choice = random.randint(0, len(buf)-1)
                    buf[-1],buf[choice] = buf[choice],buf[-1]
                    yield buf.pop()
            except StopIteration:
                random.shuffle(buf)
                yield from buf
                return

dataloader = DataLoader(Dataset(), batch_size=args.batch_size)

# load the feature generator
tokenizer = transformers.AutoTokenizer.from_pretrained(args.feature_generator)
feature_generator = transformers.AutoModel.from_pretrained(args.feature_generator)

def make_features(x):
    '''
    given a list of b strings as input,
    return a tensor of shape [b,768] that represents the features of the strings
    '''
    encoding = tokenizer.batch_encode_plus(
        x,
        max_length = 64,
        truncation = True,
        pad_to_max_length = True,
        return_tensors = 'pt',
        )
    with torch.no_grad():
        last_layer,embedding = feature_generator(**encoding) 
    last_layer.to(device)
    features = torch.mean(last_layer,dim=1)
    return features

# define the hypothesis class
class FactoredLinear(torch.nn.Module):
    '''
    this is the hypothesis class we introduced in the SGD IV notes
    for improving speed/statistical performance when k is large
    '''
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(768,args.e)
        self.linear2 = torch.nn.Linear(args.e,len(class_labels))

    def forward(self,x):
        out = self.linear1(x)
        # NOTE:
        # the following line would convert this from a linear model into a neural network,
        # the neural network has a dramatically higher VC dimension (d*e + e*k),
        # and so requires significantly more data to train
        # out = torch.nn.functional.relu(out)
        out = self.linear2(out)
        return out

h = FactoredLinear()
h.to(device)

# define our loss functions
criterion = torch.nn.CrossEntropyLoss()
reg = lambda h: sum([torch.norm(w)**2 for w in h.parameters()])

# define the optimizer
# NOTE:
# this uses the "ugly" python trick based on eval 
# to automatically define the optimizer from the input arguments
# and supporting all possible optimizers supported in pytorch
optimizer = eval('torch.optim.'+args.optimizer)(h.parameters(), lr=args.eta)

# training loop
for t, batch in enumerate(dataloader):
    logging.info("t="+str(t))

    if args.mode=='feature_generation':
        features = make_features(batch['text'])
        with open('features2.jsonl','x') as f:
            for i in range(args.batch_size):
                data = json.dumps({
                    'screen_name':batch['screen_name'][i],
                    'features':list(map(float,list(features[i].cpu().numpy()))),
                    'text':batch['text'][i]
                    })
                f.write(data+'\n')

    elif args.mode=='train':
        if 'features' not in batch:
            batch['features'] = make_features(batch['text'])

        # compute the derivative of h
        optimizer.zero_grad()
        output = h(batch['features'])
        target = batch['label_id']
        loss = criterion(output, target) 
        f = loss + args.lambda_*reg(h)
        f.backward()

        # panic on nan
        if torch.isnan(loss).any():
            logging.error('loss is nan')
            sys.exit(1)

        # call the optimizer
        optimizer.step() 

        # log to tensorboard
        norm = sum([torch.norm(w) for w in h.parameters()])
        writer.add_scalar('optimization/norm(v_t)', norm, t)
        writer.add_scalar('optimization/loss', loss, t)
        writer.add_scalar('optimization/f', f, t)

        # calculate accuracy@k
        # FIXME:
        # implement this
