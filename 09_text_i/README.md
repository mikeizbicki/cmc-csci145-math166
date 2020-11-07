# Text I: Transformers

We will apply the theory we have learned so far to text classification tasks.
In the first part, we will use the [transformers library](https://huggingface.co/transformers/) to generate features from text for us.

Transformers is a technique from deep learning that is relatively easy to use, and gives state of the art classification accuracies.
As we will see, it is also very resource intensive (both time and memory),
and provides little insight into the data.

## Homework

**Due:** Wednesday, 11 Nov midnight

The `train.py` file trains a model for predicting which congressional representative sent a tweet given only its text.

~~You should download the training data from either:~~
1. ~~https://github.com/alexlitel/congresstweets (the data is in the `/data` folder; it is about 1GB of data, but does not contain any features, and is therefore slow to run)~~
1. ~~https://134.173.191.241:8000/congresstweets.jsonl (the file should be saved into a folder named `data-features`; it is about 20GB, but does contain the feature vectors, and is therefore fast to run)~~

**Update:**
The training data is now included in the github repo in the folder `congresstweets/data_filtered`.
I've modified the training data so that it contains only the twitter accounts for the 10 most active congress members on twitter
(according to [this list](https://www.washingtonpost.com/graphics/2019/lifestyle/magazine/amp-stories/twitter/).
Since most of these congress members use multiple twitter accounts,
this means there are now 22 class labels instead of the full 1400.
The dataset has therefore been reduced from >1Gb down to 60Mb.

Your tasks:

1. Modify the `train.py` file so that it records the accuracy@1, accuracy@3, accuracy@5, accuracy@10, and accuracy@22 metrics ~~accuracy@1, accuracy@5, accuracy@10, and accuracy@100 metrics~~
1. Run the training code using the command
   ```
   $ python3 train.py --e=2 --eta=1e-2 --lambda=1e-4 --batch_size=64 --datapath=congresstweets/data_filtered
   ```
   **NOTE:**
   The command above defaults to using the `t5-small` feature generator,
   which is relatively fast but inaccurate.
   Some students have reported trouble with this model.
   If you are in that situation, then add the `--feature-generator=albert-base-v2` to use a different slightly slower but more accurate model.
1. Open the results in tensorboard, take a screenshot of your accuracies, and upload them to sakai.
   You must show at least 1k iterations of SGD.
   (This took 2 hours on my laptop.)

