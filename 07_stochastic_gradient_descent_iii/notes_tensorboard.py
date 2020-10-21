#!/usr/bin/python3

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from torch.utils.tensorboard import SummaryWriter


writer = SummaryWriter('tensorboardexample/example3')

writer.add_scalar('category/plot', 0, 0)
writer.add_scalar('category/plot', 1, 1)
writer.add_scalar('category/plot', 2, 2)
writer.add_scalar('category/plot', 3, 3)

for i in range(10):
    writer.add_scalar('category/plot2', -i, i)

for i in range(1000):
    import math
    writer.add_scalar('math/sin', math.sin(i/50), i)

for i in range(1000):
    import math
    writer.add_scalar('math/sin2', math.sin(i/5)*10+i/50, i)
