#!/usr/bin/python3

# process command line arguments
import argparse
parser = argparse.ArgumentParser()

parser_options = parser.add_argument_group('options')
parser_options.add_argument('--datapath_train',required=True)
parser_options.add_argument('--datapath_test',required=True)
parser_options.add_argument('--class_labels',default='class_labels.json')

parser_hyperparam = parser.add_argument_group('hyperparameters')
parser_hyperparam.add_argument('--eta', type=float, required=True)
parser_hyperparam.add_argument('--e', type=int, default=2)
parser_hyperparam.add_argument('--l1', type=float, default=0.0)
parser_hyperparam.add_argument('--l2', type=float, default=0.0)
parser_hyperparam.add_argument('--ngrams', type=int, default=1)
parser_hyperparam.add_argument('--optimizer',default='SGD')
parser_hyperparam.add_argument('--batch_size',type=int,default=64)
parser_hyperparam.add_argument('--max_epochs',type=int,default=100)

parser_features = parser.add_argument_group('features')
parser_features.add_argument('--feature_generator',default='1hot')
parser_features.add_argument('--feature_d_exp',type=int,default=10)

parser_debug = parser.add_argument_group('debug')
parser_debug.add_argument('--logdir', default='log')
parser_debug.add_argument('--seed', default=0)
parser_debug.add_argument('--device',choices=['auto','cpu','gpu'],default='auto')
parser_debug.add_argument('--ks',nargs='+',default=[1, 3, 5, 10, 22])
args = parser.parse_args()

# load libraries
from collections import Counter
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
torch.set_num_threads(1)

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
logdir = os.path.join(
    args.logdir,
    'feature_generator='+args.feature_generator+
    ',feature_d_exp='+str(args.feature_d_exp)+
    ',optimizer='+args.optimizer+
    ',eta='+str(args.eta)+
    'batch_size='+str(args.batch_size)+
    ',e='+str(args.e)+
    ',l2='+str(args.l2)+
    ',l1='+str(args.l1)+
    ',ngrams='+str(args.ngrams)
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
    def __init__(self, datapath):
        super().__init__()
        self.datapath = datapath

    def __iter__(self):
        return Dataset.scramble(Dataset.yield_data(self.datapath), 8096)

    def yield_data(datapath):
        '''
        creates a generator that yields a single tweet at a time;
        the results are yielded in the order they are present on disk,
        and so the results must be combined with the scramble function 
        to generate a random sample
        '''
        filenames = sorted(os.listdir(datapath))
        random.shuffle(filenames)
        for filename in filenames:
            filepath = os.path.join(datapath,filename)
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
        randomizes the order of a generator using O(1) memory instead of O(n);
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

dataloader_train = DataLoader(Dataset(args.datapath_train), batch_size=args.batch_size)
dataloader_test = DataLoader(Dataset(args.datapath_test), batch_size=args.batch_size)

# load the feature generator
if args.feature_generator == '1hot':
    import spacy
    import torch
    nlp = spacy.load('en_core_web_sm',  disable=["parser", "ner"])
    d = 2**args.feature_d_exp

    def make_features_batch(xs, prime=19134702400093278081449423917, seed=0):

        indexes = []
        values = []

        docs = list(nlp.pipe(xs))

        for i,doc in enumerate(docs):

            # compute the lemmas
            lemmas = []
            grams = []
            tokens = list(doc)
            for n in range(1,args.ngrams+1):
                for j in range(len(doc)-n+1):
                    gram = tokens[j:j+n]
                    grams.append(' '.join([ token.lemma_ for token in gram]))
                    lemma_sum = sum([ token.lemma for token in gram])
                    lemmas.append((lemma_sum+seed)*prime%d)
            lemmas.sort()

            # compute indexes and values for x
            indexes_x = []
            values_x = []
            if len(lemmas) > 0:
                indexes_x.append(lemmas[0])
                values_x.append(1)
            for lemma in lemmas[1:]:
                if lemma==lemmas[-1]:
                    values_x[-1] += 1
                else:
                    indexes_x.append(lemma)
                    values_x.append(1)

            # add to the batch
            indexes.extend([[i,index] for index in indexes_x])
            values.extend(values_x)

        return torch.sparse.FloatTensor(
                torch.LongTensor(indexes).t(),
                torch.FloatTensor(values),
                torch.Size([len(xs),d])
                )

else:
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.feature_generator)
    feature_generator = transformers.AutoModel.from_pretrained(args.feature_generator)
    def make_features_batch(x):
        '''
        given a list of b strings as input,
        return a tensor of shape [b,feature_generator_dimensions] 
        that represents the features of the strings
        '''
        encoding = tokenizer.batch_encode_plus(
            x,
            max_length = 64,
            truncation = True,
            pad_to_max_length = True,
            return_tensors = 'pt',
            )
        with torch.no_grad():
            del encoding['token_type_ids']
            last_layer,embedding = feature_generator(**encoding) 
        last_layer.to(device)
        features = torch.mean(last_layer,dim=1)
        return features
    d = make_features_batch(['this is a test']).shape[1]


# define the hypothesis class
class FactoredLinear(torch.nn.Module):
    '''
    this is the hypothesis class we introduced in the SGD IV notes
    for improving speed/statistical performance when k is large
    '''
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(d,args.e)
        self.fc2 = torch.nn.Linear(args.e,len(class_labels))

    def forward(self,x):
        out = self.fc1(x)
        out = self.fc2(out)
        return out

h = FactoredLinear()
torch.nn.init.constant_(h.fc1.weight, 0)
torch.nn.init.constant_(h.fc1.bias, 0)
h.to(device)

# define our loss functions
criterion = torch.nn.CrossEntropyLoss()
l2reg = lambda h: sum([torch.norm(w,2)**2 for w in h.parameters()])
l1reg = lambda h: torch.norm(h.fc1.weight,1)

# define the optimizer
# NOTE:
# this uses the "ugly" python trick based on eval 
# to automatically define the optimizer from the input arguments
# and supporting all possible optimizers supported in pytorch
optimizer = eval('torch.optim.'+args.optimizer)(h.parameters(), lr=args.eta)

# training loop
step = 0
for epoch in range(args.max_epochs):
    logging.info("epoch="+str(epoch))

    # complete a training epoch
    for t, batch in enumerate(dataloader_train):
        step += 1
        logging.info("train t="+str(t))

        # generate the features if they're not available
        if 'features' not in batch:
            batch['features'] = make_features_batch(batch['text'])

        # compute the derivative of the optimization objective with respect to h
        optimizer.zero_grad()
        output = h(batch['features'])
        target = batch['label_id']
        loss = criterion(output, target) 
        l2 = l2reg(h)
        l1 = l1reg(h)
        f = loss + args.l2*l2 + args.l1*l1
        f.backward()

        # panic on nan
        if torch.isnan(loss).any():
            logging.error('loss is nan')
            sys.exit(1)

        # call the optimizer
        optimizer.step() 

        # log to tensorboard
        norm = sum([torch.norm(w) for w in h.parameters()])
        writer.add_scalar('optimization/norm(v_t)', norm, step)
        writer.add_scalar('optimization/loss', loss, step)
        writer.add_scalar('optimization/f', f, step)
        writer.add_scalar('optimization/l1', l1, step)
        writer.add_scalar('optimization/l2', l2, step)

        for threshold in [1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]:
            nnz_fc1 = (torch.abs(h.fc1.weight)>threshold).sum()
            nnz_fc2 = (torch.abs(h.fc2.weight)>threshold).sum()
            writer.add_scalar('nnz_fc1/'+str(threshold), nnz_fc1, step)
            writer.add_scalar('nnz_fc2/'+str(threshold), nnz_fc2, step)

        # calculate accuracy@k
        top_n, top_i = output.topk(max(args.ks))
        for k in args.ks:
            acc_k = torch.sum(top_i[:,:k]==target.unsqueeze(1))/float(top_i.shape[0])
            writer.add_scalar('acc/@'+str(k), acc_k, step)

    # save the model
    torch.save(h.state_dict(), os.path.join(logdir,'model-epoch='+str(epoch)))

    # compute the accuracies on the test set
    test_acc = Counter()
    test_dp = 0
    for t, batch in enumerate(dataloader_test):
        logging.info("test t="+str(t))
        with torch.no_grad():

            # generate the features if they're not available
            if 'features' not in batch:
                batch['features'] = make_features_batch(batch['text'])

            # compute accuracies for the batch
            output = h(batch['features'])
            target = batch['label_id']
            top_n, top_i = output.topk(max(args.ks))
            for k in args.ks:
                test_acc[k] += torch.sum(top_i[:,:k]==target.unsqueeze(1))
            test_dp += target.shape[0]

    # write the accuracies to tensorboard
    for k in args.ks:
        writer.add_scalar('test_acc/@'+str(k), test_acc[k]/test_dp, epoch)

    # write class embeddings to tensorboard
    writer.add_embedding(
        mat = h.fc2.weight,
        metadata = class_labels,
        global_step = epoch,
        tag = 'twitter account embedding'
        )
