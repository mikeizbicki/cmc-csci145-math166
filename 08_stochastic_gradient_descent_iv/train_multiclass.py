#!/use/bin/python3

# process command line arguments
import argparse
parser = argparse.ArgumentParser()

parser_data = parser.add_argument_group('dataset')
parser_data.add_argument('--d',default=20)
parser_data.add_argument('--m',default=200)
parser_data.add_argument('--k',default=100)
parser_data.add_argument('--m_test',default=2000)

parser_hyperparam = parser.add_argument_group('hyperparameters')
parser_hyperparam.add_argument('--eta', type=float, required=True)
parser_hyperparam.add_argument('--T', type=int, required=True)
parser_hyperparam.add_argument('--lambda', dest='lambda_', type=float, default=0.0)
parser_hyperparam.add_argument('--algorithm',choices=['sgd','gd'],required=True)

parser_debug = parser.add_argument_group('debug')
parser_debug.add_argument('--logdir', default='log')
parser_debug.add_argument('--seed', default=0)
args = parser.parse_args()

# load libraries
import os
import random
import sys
import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    )
import torch
torch.manual_seed(args.seed)

# generate directory for tensorboard logs
logdir = os.path.join(
    args.logdir,
    'k='+str(args.k)+',d='+str(args.d)+',m='+str(args.m)+',algorithm='+args.algorithm+',eta='+str(args.eta)+',T='+str(args.T)+',lambda='+str(args.lambda_)
    )
try:
    os.makedirs(logdir)
except FileExistsError:
    logging.error('logdir='+logdir+' already exists')
    import sys
    sys.exit(1)

# load tensorboard
# NOTE:
# tensorboard is part of the tensorflow library, not pytorch;
# tensorflow has an annoying habbit of printing tons of debugging information;
# the os.environ line disables printing this information before performing the import
# see: https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(logdir)

# load the data
import sklearn.datasets
X,Y = sklearn.datasets.make_classification(
        n_samples=args.m+args.m_test,
        n_features=args.d,
        n_informative=args.d,
        n_redundant=0,
        n_repeated=0,
        random_state=args.seed,
        n_classes=args.k
        )
X = torch.Tensor(X)
Y = torch.tensor(Y)

X_train = X[:args.m]
Y_train = Y[:args.m]
X_test  = X[args.m:]
Y_test  = Y[args.m:]
logging.info("X_train.shape="+str(X_train.shape))
logging.info("Y_train.shape="+str(Y_train.shape))
logging.info("X_test.shape=" +str(X_test.shape ))
logging.info("Y_test.shape=" +str(Y_test.shape ))

# define the hypothesis class
# NOTE:
# the name hypothesis "class" comes from set theory,
# where a class is a special type of "higher order set" designed 
# to avoid set paradoxes like "the set of all sets that don't contain themselves";
# the "class" from programming derives from this set-theoretic origin;
# a class is a special type of set, like "int", "float", and "tensor"
class LinearModel(torch.nn.Module):

    # NOTE:
    # the __init__ function defines all of parameters for the hypothesis;
    # that is, it defines everything to the "right of the colon"
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.Tensor(torch.ones([args.d,args.k])))

    # NOTE:
    # the forward function defines how the hypothesis computes a value using the parameters;
    # that is, it defines everything to the "left of the colon"
    # (but traditionally does not include the loss function)
    def forward(self, x):
        out = x @ self.w
        return out

    # NOTE:
    # the torch.nn.Module class contains the following definition
    '''
    def __call__(self, x):
        return self.forward(x)
    '''

h = LinearModel()
param_sizes = [ param.size() for param in h.parameters() ]
logging.info('param_size='+str(param_sizes))
#logging.info('h.parameters()='+str(list(h.parameters())))

# NOTE:
# linear models are a fundamental data mining tool,
# and so the hypothesis class of linear models is built-in to pytorch
'''
h = torch.nn.Linear(in_features=[args.d], out_features=[args.k], bias=False)
'''

# define our loss functions
# NOTE:
# in practice, we almost always use loss functions that are built-in to pytorch;
# this is for both numeric stability and speed reasons
criterion = torch.nn.CrossEntropyLoss()
reg = lambda h: sum([torch.norm(w)**2 for w in h.parameters()])

# define the optimizer
# NOTE:
# this optimizer is called SGD, but it actually implements both SGD and GD;
# recall that both of these optimizers have the same update equations,
# they just use a different value for the gradient;
# by choosing our loss function below, we choose which algorithm will be employed;
optimizer = torch.optim.SGD(h.parameters(), lr=args.eta)

# NOTE:
# there are many other optimizers implemented in torch as well;
# for a full list, see: https://pytorch.org/docs/stable/optim.html
# uncommenting the line below will enable the Adam optimizer;
# Adam and SGD are the two most popular optimizers used in practice;
# for details of the adam optimizer, see: https://arxiv.org/abs/1412.6980
'''
optimizer = torch.optim.Adam(h.parameters(), lr=args.eta)
'''

# NOTE:
# What's the difference between SGD and the other optimizers?
# SGD has proofs bounding both the *generalization error* L_S(w_t) - L_D(w_t)
# and the training error L_S(w_t) - L_S(w^*);
# the other optimizers have proofs bounding only the training error L_S(w_t) - L_S(w^*),
# and the generalization error can be arbitrarily bad
# See: https://arxiv.org/abs/1705.08292

# training loop
for t in range(args.T):
    logging.info("t="+str(t))

    # compute the derivative of h
    optimizer.zero_grad()
    if args.algorithm=='gd':
        # FIXME:
        # gradient descent requires using the L_S loss function over the entire data set
        # instead of the sgd loss function defined below over a single data point;
        raise ValueError('gd not implemented')
    elif args.algorithm=='sgd':
        i = random.randint(0,args.m)
        #i = t%args.m
        output = h(X[i])
        target = Y[i]
        output = output.reshape([1,args.k])
        target = target.reshape([1])
        loss = criterion(output, target) + args.lambda_*reg(h)
    loss.backward()

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
