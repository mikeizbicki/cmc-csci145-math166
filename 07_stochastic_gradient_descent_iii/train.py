#!/use/bin/python3

# process command line arguments
import argparse
parser = argparse.ArgumentParser()

parser_data = parser.add_argument_group('dataset')
parser_data.add_argument('--d',default=20)
parser_data.add_argument('--m',default=200)
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
    'd='+str(args.d)+',m='+str(args.m)+',algorithm='+args.algorithm+',eta='+str(args.eta)+',T='+str(args.T)+',lambda='+str(args.lambda_)
    )
try:
    os.makedirs(logdir)
except FileExistsError:
    logging.error('logdir='+logdir+' already exists')
    import sys
    sys.exit(1)

# load tensorboard
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
        )
X = torch.Tensor(X)
Y = torch.Tensor(Y)

X_train = X[:args.m]
Y_train = Y[:args.m]
X_test  = X[args.m:]
Y_test  = Y[args.m:]
logging.info("X_train.shape="+str(X_train.shape))
logging.info("Y_train.shape="+str(Y_train.shape))
logging.info("X_test.shape=" +str(X_test.shape ))
logging.info("Y_test.shape=" +str(Y_test.shape ))

# define the hypothesis class
w_t = torch.Tensor(torch.ones([args.d]))
ws = []

# define our loss functions
logistic_loss = lambda a: torch.log(1 + torch.exp(-a))
reg = lambda w: torch.norm(w)**2

mean = lambda xs: sum(xs)/len(xs)

L_S = lambda w: mean([ logistic_loss(Y_train[i] * X_train[i,:].t() @ w_t) for i in range(args.m) ])
L_D = lambda w: mean([ logistic_loss(Y_test [i] * X_test [i,:].t() @ w_t) for i in range(args.m_test) ])

# training loop
for t in range(args.T):
    logging.info("t="+str(t))

    # compute the derivative of w_t
    w_t.requires_grad = True
    if args.algorithm=='gd':
        loss = L_S(w_t) + args.lambda_*reg(w_t)
    elif args.algorithm=='sgd':
        loss = logistic_loss(Y[t%args.m] * X[t%args.m,:].t() @ w_t) + args.lambda_*reg(w_t)
    loss.backward()

    # panic on nan
    if torch.isnan(loss).any():
        logging.error('loss is nan')
        sys.exit(1)

    with torch.no_grad():

        # perform the gradient update steps
        v_t = w_t.grad
        w_t = w_t - args.eta * v_t

        # compute wbar
        ws.append(w_t)
        wbar = (1/len(ws)) * sum(ws)
        loss_wbar = L_S(wbar)

        # log to tensorboard
        writer.add_scalar('optimization/norm(v_t)', torch.norm(v_t), t)
        writer.add_scalar('optimization/loss', loss, t)

        writer.add_scalar('results/L_S(w_t)' , L_S(w_t) , t)
        writer.add_scalar('results/L_S(wbar)', L_S(wbar), t)
        writer.add_scalar('results/L_D(w_t)' , L_D(w_t) , t)
        writer.add_scalar('results/L_D(wbar)', L_D(wbar), t)
