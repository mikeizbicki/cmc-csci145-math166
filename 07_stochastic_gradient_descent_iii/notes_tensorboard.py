#!/usr/bin/python3
'''
Tensorboard is a library for generating plots that can be used to debug data mining algorithms.
This short note demonstrates how to create plots with tensorboard.
'''

import math

# disable excessive warning messages caused by tensorflow
# see: https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# load pytorch's interface into the tensorboard library
# NOTE:
# tensorboard is technically part of the tensorflow library,
# which is Google's version of the pytorch library
# NOTE:
# the parameter to SummaryWriter is a path,
# and it must have exactly two folders inside of it
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('set_of_experiments/experiment_name')

# create an example plot
# NOTE:
# writer.add_scalar is the main method for creating plots
# parameter 1: plot name
# parameter 2: x value
# parameter 3: y value
writer.add_scalar('category/plot', 1, 0)
writer.add_scalar('category/plot', 1, 1)
writer.add_scalar('category/plot', 1, 2)
writer.add_scalar('category/plot', 1, 3)

# create more example plots
for i in range(10):
    writer.add_scalar('category/plot2', -i, i)

for i in range(1000):
    writer.add_scalar('math/sin', math.sin(i/50), i)

for i in range(1000):
    writer.add_scalar('math/sin2', math.sin(i/5)*10+i/50, i)
