# VowpalWabbit

Vowpal Wabbit is an extremely fast machine learning system maintained by [John Langford](https://www.microsoft.com/en-us/research/people/jcl/publications/).
VW is [widely used in industry](https://github.com/mritunjaysharma394/vowpal-wiki/blob/master/Awesome-Vowpal-Wabbit.md),
and gives SOTA results for "simple" text based problems.
For example:

1. Choosing which ads to display to a user.

1. Choosing which articles to display for a user on a newspaper's webpage.

1. Choosing which messages to display on social media.

Getting good performance with Vowpal Wabbit requires a deep understanding of VC theory in order to tune hyperparameters correctly.
This purpose of this assignment is to give you practice applying all the theory we've learned in class to a real world problem.

> **Note:**
>
> More "complex" text problems require deep learning,
> and these problems are commonly solved using transfer learning.
> It's not exactly clear where the dividing line is between "simple" and "complex",
> but in practice it's always a good idea to start with Vowpal Wabbit because it is so easy to use and fast.
> Only switch to deep learning if Vowpal Wabbit isn't good enough for your application.

> **Note:**
>
> Vowpal Wabbit is extremely famous among machine learning professionals,
> but not as famous among the general public as tools like pytorch/tensorflow.
> Personally, I think it's a __*MUCH*__ more valuable tool in practice,
> but it's less well known because you actually need to understand machine learning theory in order to be able to use it well.

> **Note:**
>
> In this assignment, we will use Vowpal Wabbit's [command line interface](https://vowpalwabbit.org/docs/vowpal_wabbit/python/latest/tutorials/cmd_first_steps.html).
> (And you will need to do this on the lambda server since that's where the data is.)
> For people experienced with the command line, this is by far the easiest way to use VW, and the easiest way to train models in general.
> VW also has a [python interface compatible with scikit learn](https://vowpalwabbit.org/docs/vowpal_wabbit/python/latest/reference/vowpalwabbit.sklearn.html).

> **Fun Aside:**
>
> Like all good machine learning software, it is open source.
> Unlike most open source software, it even has contributions from [North Koreans](https://izbicki.me/blog/teaching-open-source-in-north-korea.html) in the source code.

<!--
1. Vowpal Wabbit command line tutorial: <https://vowpalwabbit.org/docs/vowpal_wabbit/python/latest/tutorials/cmd_linear_regression.html>
-->

## Your Task

You are going to create a machine learning model that determines the political bias (pro-democrat or pro-republican) of a piece of text.

**The Data:**

Gathering labelled data for this task is potentially difficult and expensive.
The most obvious way to hire people to read political text and label it manually.
This has several problems:

1. It's expensive to generate labels.
1. The labellers may not agree about which texts are pro-democrat or pro-republican.
1. It can be hard to decide what text we should label, and our choice of text will strongly impact the results.

Fortunately, there is "naturally" labeled text in the form of congressional tweets.
The github repo <https://github.com/alexlitel/congresstweets> contains all tweets sent by members of congress since 2017-06-21.
We will assume that tweets written by democrats are pro-democrat and tweets written by republicans are pro-republican.
This solves all the problems of manually labeling text above.

I've divided the data into a training and a test set and converted the data into Vowpal Wabbit's data format.
You can take a look at the training file with the command
```
$ head /data-fast/congresstweets/data_2017-2021 
1 | The Affordable Care Act should be improved But the answer is not to throw 23 million Americans off of health insurance
```
The number to the left of the `|` is the class label.
In this dataset, a `1` indicates it was sent by a democrat and a `-1` indicates it was sent by a republican.
The text after the `|` is the text of the tweet that has been processed to have punctuation marks removed.

The training set consists of all tweets sent before 2022,
and the test set consists of all tweets sent during 2022 (up to 20 November).
It is common when working with social media data to split the training/test set by date;
this helps ensure that our model will generalize to new,
upcoming years and isn't specialized to just a specific time period.

You can count the number of tweets in the training set with the command
```
$ wc -l /data-fast/congresstweets/data_2017-2021
134246
```
and the test set with
```
$ wc -l /data-fast/congresstweets/data_2022
20696
```

<!--
The most difficult part about the congresstweets dataset is figuring our which twitter accounts correspond to republicans, and which correspond to democrats.
(Many congress members have multiple accounts, and there's more than 1000 accounts in total in the dataset.)
So I've selected only the 10 most active Twitter users and generated a dataset from them (6 democrats and 4 republicans).
Details are available in the `/data-fast/congresstweets/filter_dataset.py` file, but you don't need to review the details.

I've also divided the data into 
-->

**Preliminary Task:**

This first task ensures that you have a sane working environment.
There is nothing to turn in for this task, but you need to be able to successfully run these commands in order to complete the main task below.

On the lambda server, use vowpal wabbit to train a simple model with the following command:

```
$ vw -d /data-fast/congresstweets/data_2017-2021 -f model.vw --loss_function=logistic --binary
```
The meaning of this command is:
1. `vw` is the name of the Vowpal Wabbit command.
1. `-d /data-fast/congresstweets/data_2017-2021` specifies the data file to train on.
1. `-f model.vw` specifies the name of the file to save the model weights into.
1. `--loss_function=logistic` specifies to use logistic regression.
(The default is ordinary least squares.)
1. `--binary` causes the 0-1 loss to be printed instead of the logistic loss.

The output of the command should look something like
```
Num weight bits = 18
learning rate = 0.5
initial_t = 0
power_t = 0.5
using no cache
Reading datafile = /data-fast/congresstweets/data_2017-2021
num sources = 1
average  since         example        example  current  current  current
loss     last          counter         weight    label  predict features
1.000000 1.000000            1            1.0   1.0000   0.0000       22
0.884311 0.768623            2            2.0   1.0000   0.1233       24
0.698704 0.513096            4            4.0   1.0000   0.3308       21
0.527718 0.356732            8            8.0   1.0000   0.2119       22
0.325783 0.123848           16           16.0   1.0000   0.8986       27
0.246375 0.166967           32           32.0   1.0000   0.8712       28
0.178432 0.110490           64           64.0   1.0000   0.7311       23
0.156420 0.134408          128          128.0   0.0000   0.5785       21
0.127104 0.097787          256          256.0   1.0000   1.0000       62
0.091761 0.056419          512          512.0   1.0000   0.5251       10
0.103440 0.115119         1024         1024.0   0.0000   0.4586       22
0.096746 0.090052         2048         2048.0   1.0000   1.0000       63
0.090744 0.084742         4096         4096.0   0.0000   0.0000       17
0.082233 0.073721         8192         8192.0   1.0000   1.0000       26
0.073999 0.065766        16384        16384.0   1.0000   1.0000       20
0.059888 0.045776        32768        32768.0   1.0000   1.0000       90
0.051809 0.043731        65536        65536.0   1.0000   1.0000       60
0.050859 0.049910       131072       131072.0   1.0000   1.0000       99

finished run
number of examples per pass = 134246
passes used = 1
weighted example sum = 134246.000000
weighted label sum = 80568.000000
average loss = 0.050966
best constant = 0.600152
best constant's loss = 0.239970
total feature number = 6422170
```

You see the predictions of your model by running the command
```
$ vw -i model.vw -p /dev/stdout --binary --quiet <<< '| The Affordable Care Act should be improved But the answer is not to throw 23 million Americans off of health insurance'
1
```
The meaning of this command is:
1. `-i model.vw` specifies the name of the model weights to load.  This must match whatever you passed to `-f` when training the model.
1. `-p /dev/stdout` says to print the results of the prediction to the terminal.
1. `--binary` causes the 0-1 loss to be printed instead of the logistic loss.
1. `--quiet` suppresses debugging information.
1. `<<<` is input redirection from the following string.

The prediction is printed to the terminal.
In this case, the value `1` indicates that the model strongly thinks the tweet was sent by a democrat,
and this tweet is in fact the first tweet sent by a democrat in our training data.

Similarly, if we evaluate the first tweet sent by a republican, we get a very strong prediction that it was sent by a republican:
```
$ vw -i model.vw -p /dev/stdout --binary --quiet <<< '| SOON I will be delivering remarks at the Institute of Caribbean Studies Tune in here t co gX17XZFcy9'
-1
```

<!--
Our model is not perfect, however.
The following tweet was sent by a republican (Senator Rubio),
but our model has no idea who sent it:
```
$ vw -i model.vw -p /dev/stdout --quiet <<< '| It is really critical that people have confidence that when they go vote that vote is going to count t co OG4QHHJTef'
0.500648
```
-->

Next we see how to measure our model's performance with the command
```
$ vw -d /data-fast/congresstweets/data_2022 -i model.vw -t --binary
```
One again:
1. `-d /data-fast/congresstweets/data_2022` specifies the data set to test on
1. `-i model.vw` specifies the name of the model weights to load.  This must match whatever you passed to `-f` when training the model.
1. `-t` specifies that you are testing and not training the model.
1. `--binary` causes the 0-1 loss to be printed instead of the logistic loss.

There will be lots of output, but the important line is
```
average loss = 0.089293
```
which shows your testing loss.

**Task:**

Find a set of hyperparameters that you can pass to Vowpal Wabbit so that a model trained on `data_2017-2021` and tested on `data_2022` has testing loss less than 0.08.

Upload to Sakai:

1. The command you used to train your model, and its output

1. The command you used to test your model, and its output

**Extra Credit:**

You can earn up to 2% extra credit on your final grade in this class if you achieve a test set accuracy less than 0.075.
In order to receive the extra credit,
you will have to explain to me during your final exam how you achieved this set of hyperparameters.
You will not receive the extra credit if you just "copy" the values from someone else and cannot explain to me the procedure you used to find them.

