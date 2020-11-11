# Text II Homework

**Due:** Wednesday, 18 Nov midnight

Like before, the `train.py` file trains a model for predicting which congressional representative sent a tweet given only its text.

I have divided the training data into two sets, contained within the `data_train` folder (all tweets sent before 2020) and the `data_test` folder (all tweets sent after 2020).
Your goal is to find a set of hyperparameters that achieves an accuracy@1 score >= 0.54 and accuracy@3>= 0.75 on the test set.
You will do this by running the command
```
$ python3 train.py --datapath_train=data_train --datapath_test=data_test $hyperparams
```
where `$hyperparams` is the set of hyperparameters you have chosen.

Recall that:

1. `--optimizer` can be set to either `SGD`, `Adam`, or any other optimizer supported by pytorch.
   Selecting `SGD` will give you better generalization performance,
   but selecting `Adam` will give you much faster convergence, especially with l1 regularization.
   I recommend experimenting with both to see how their behavior is different.
   (It's possible to achieve good results with either,
   but your other hyperparameters will be very different for each choice.)

1. `--eta` primarily affects your computational performance and has negligible effect on statistical performance.
   As long as `--eta` is "small enough",
   then the optimization is guaranteed to converge to a global minimum.
   You should try to find the largest value of `--eta` that does not cause your model to diverge.
   Values between 1e0 and 1e-6 are reasonable.

1. `--feature_generator` affects both the computational and statistical performance of your model.
   The `1hot` feature generator is significantly faster than any of the transformer models,
   so I recommend using it.
   The transformer models have a lower bayes and approximation error,
   but a much higher estimation error.

1. `--feature_d_exp` affects both the computational and statistical performance of your model.
   The number of feature dimensions is `2**feature_d_exp`, and so increasing `--feature_d_exp` will result in large vectors, increasing memory usage and the runtime of matrix multiplication.
   The larger parameter vector results in fewer hash collisions, reducing bayes error.
   The larger parameter vectors result in an increased VC-dimension [1], increasing estimation error.
   The larger parameter vectors also increase the `B` parameter in the SGD theorems,
   resulting in a larger generalization error.
   Any value between 10 and 30 is reasonable.

1. `--ngrams` affects both the computational and statistical performance of your model.
   Increasing `--ngrams` will reduce the sparsity of your feature vectors (increasing nnz).
   The speed of matrix multiplication is proportional to nnz,
   so the matrix multiplications will take longer.
   The `rho` value in the SGD theorems is also proportional to nnz,
   so larger `--ngrams` results in larger generalization error.
   Increasing `--ngrams` will reduce the bayes error.
   Any value between 1 and 10 is reasonable.

1. `--e` affects both the computational and statistical performance of your model.
   Increasing `--e` causes your matrices to increase in size, resulting in more memory usage and longer runtimes for matrix multiplications.
   Increasing `--e` increases the VC-dimension [1] of your model,
   reducing approximation error and increasing estimation error.
   The `--e` parameter is proportional to the `B` value from the SGD theorems,
   so large `--e` values result in larger generalization error.
   Any value between 1 and the number of classes (22) is reasonable.

1. `--l1` affects the statistical performance of your solution.
   Increasing `--l1` induces sparsity.
   This "effectively" reduces the VC-dimension [1] of your model,
   increasing approximation error and reducing estimation error.
   Large values of `--l1` also increase your model's bias and reduce the model's variance.
   Values between 1e0 and 1e-6 are reasonable.

1. `--l2` affects the computational and statistical performance of your solution.
   Large values of `--l2` increase the strong convexity of your model, resulting in faster convergence to the global minimum.
   Large values of `--l1` also increase your model's bias and reduce the model's variance.
   Values between 1e0 and 1e-6 are reasonable.

1. `--batch_size` affects the computational performance of your model.
   Modern hardware can parallelize certain parts of the matrix multiplication algorithms,
   and setting the batch size to between 8-256 will help take advantage of this.
   The optimal value will depend on your computer's particular hardware configuration,
   but it's probably not worth spending anytime optimizing this value.
   The default value of 64 should work pretty well for everyone.
   No matter what your batch size is, you are guaranteed to converge to the same solution, so there is no affect on statistical performance.

1. `--epochs` affects both the computational and statistical performance of your model.
   Increasing the number of epochs makes the code take longer to run.
   Decreasing the number of epochs means that you may not converge to the global optimum.
   I recommend using a large value (e.g. 100), and then manually stopping the code (pressing CTRL+C) on runs that do not seem promising.

[1] Technically, we're in a multi-class setting,
so it should be the "Natarajan" dimension instead of the VC-dimension,
but it's essentially the same thing,
with the same idea of approximation/estimation errors. 
If we were to restrict the problem to have only 2 classes,
then everything I said above becomes technically correct.

NOTE:

1. None of these hyperparameter options are affecting our hypothesis class (linear model vs neural network vs boosting vs etc. ; which kernel should we use?)
   To really achieve the best prediction performance, we would want to investigate those as well.
   In data mining applications, however, linear models are almost always used in practice because they are easy to interpret (e.g. https://datainsights.pub/claremont_news/).

HINT:

1. Start by making your bayes error and approximation error small.

    1. Either you'll get good results (in which case you're done), or you know your estimation error must be large.
       Then try to reduce your estimation error while making only small sacrifices on your bayes/approximation errors.

       What is a "small" sacrifice?
       That's hard to say exactly, but the theorems help give clues.

1. Because you're working on a much smaller dataset than the full congress tweets dataset,
   (number of classes has been reduced from 1400 to 22, and number of data points reduced from 10^7 down to 10^5),
   computational issues will be much less important.

1. The order of the hyperparameters above is approximately from most important at the top to least important at the bottom.

HINT:

1. You will have to run lots of experiments, so be efficient with them.
   1. Let them run in the background while you're reading/watching TV.
   1. Work in a small team where different team members try exploring different options in the search space.

HINT:

1. The state-of-the-art method for automating the process of selecting these hyperparameters for you automatically is called *random search*.
   It's "trivial" to implement
   (see https://en.wikipedia.org/wiki/Hyperparameter_optimization#Random_search and http://www.argmin.net/2016/06/23/hyperband/)
   but probably a bit more work than manually understanding the process in the first HINT.

WARNING:

1. Why is the test accuracy not a good reflection of the test accuracy you would get on 2021 data?

## Submission

Take a screenshot of the `test_acc` section of tensorboard for your best performing model.
It should have only a single run displayed with the smoothing parameter set to 0.
Upload the screenshot along with the hyperparameters you used to generate that run to sakai.
The following is an example 100% submission:

<img src=example-submission-fixed.png />

If you achieve the desired accuracy@1 and @3 scores, you will get full credit on the assignment,
otherwise you will lose points depending on how far you fall short.
