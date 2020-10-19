# SGD III: Implementation in PyTorch

<img width=80% src=tensorboard-scalars.png />

There are no notes this week for you to complete :)

Instead, we will be transition to more coding based work.

In class, we will be going over the material from the official PyTorch tutorial: https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html

## Assignment

**Due:** Sunday, 25 October at midnight

The goal of this first assignment is to ensure that you have pytorch and tensorboard working correctly together,
and to start getting a feel for how the learning rate effects your prediction results.

Steps:

1. Install tensorboard with the command
   ```
   $ pip3 install tensorboard
   ```

1. Run the `train.py` program to generate plots in tensorboard with different hyperparameters.

1. You should run 2 distinct sets of experiments, one for GD, and one for SGD.
   In each plot, you should show the hyperparameter values of setting `--eta=` `1e1`,`1e0`,`1e-1`,`1e-2`,`1e-3`,`1e-4`,`1e-5`.
   For the GD experiment, you should set `--T=20` and for the SGD experiment set `--T=2000`.
   For both experiments, set `--lambda==1e0`.

1. Take a screenshot of tensorboard for both of your experiments, and upload them to sakai.
   Tensorboard should not be showing any extraneous runs in either of the experiments.

