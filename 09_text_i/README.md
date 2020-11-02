# Text I: Transformers

We will apply the theory we have learned so far to text classification tasks.
In the first part, we will use the [transformers library](https://huggingface.co/transformers/) to generate features from text for us.

Transformers is a technique from deep learning that is relatively easy to use, and gives state of the art classification accuracies.
As we will see, it is also very resource intensive (both time and memory),
and provides little insight into the data.

## Homework

**Due:** Sunday, 8 Nov midnight

The `train.py` file trains a model for predicting which congressional representative sent a tweet given only its text.
You should download the training data from either:
1. https://github.com/alexlitel/congresstweets (the data is in the `/data` folder; it is about 1GB of data, but does not contain any features, and is therefore slow to run)
<!--
1. https://izbicki.me/public/cs/congresstweets.jsonl (the file should be saved into a folder named `data-features`; it is about 20GB, but does contain the feature vectors, and is therefore fast to run)
-->
1. https://134.173.191.241:8000/congresstweets.jsonl (the file should be saved into a folder named `data-features`; it is about 20GB, but does contain the feature vectors, and is therefore fast to run)

Your tasks:

1. Modify the `train.py` file so that it records the accuracy@1, accuracy@5, accuracy@10, and accuracy@100 metrics
1. Run the training code using the command
   ```
   $ python3 train.py --e=32 --eta=1e-2 --lambda=1e-4 --batch_size=64 --datapath=XXX
   ```
   where `XXX` is the path to the training data you've downloaded.
1. Open the results in tensorboard, take a screenshot of your accuracies, and upload them to sakai.
   You must show at least 10k iterations of SGD.

