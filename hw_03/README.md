# Homework 3: word2vec

The goal of this assignment is to use the word2vec model to learn word embeddings,
and use those word embeddings to calculate word analogies.
The assignment is divided into several parts,
and you can choose to complete whichever parts you want depending on the final grade you want.

You may complete this assignment in teams of two.

**Part 1 (60pts):**
Load a pretrained word2vec model into gensim.
The [gensim documentation](https://radimrehurek.com/gensim/models/keyedvectors.html#what-can-i-do-with-word-vectors) shows how to do this in 1 line of code on a model pretrained on 1 billion words.
If you want get really crazy, [the Stanford NLP webpage](https://nlp.stanford.edu/projects/glove/) contains a model trained on 840 billion words that takes up 2gb.

With this dataset perform two tasks:

1. Create 10 sets of word analogies (e.g. `king - man + woman = ?`) and print the output of the top 5 best words.
You should divide your set of word analogies into two sections of 5.
The first set of analogies should "work", meaning the most similar words are good choices for the result of the equation; 
the second set of analogies should "not work", meaning that the most similar words are bad choices for the equation.
The `most_similar` function in the gensim documentation above shows how to do this.

2. Create 5 sets of "one of these things is not like the other", and show that the word vectors are able to determine which of these things is not like the other.
Use the `doesnt_match` function from gensim.

**Part 2 (30 pts):**
Instead of using a pretrained model in gensim, train your own model following the gensim documentation [here](https://radimrehurek.com/gensim/models/word2vec.html).
The tutorial does not include data files, so you will have to find a large set of text to work with on your own.
One easy source of large text files is [project gutenberg](https://www.gutenberg.org/).
If you are working in a team of two, you should select 10 books from project gutenberg to train your model on; but if you are working by yourself you need to select only a single book.
When completing part 1, you should choose words that are somehow related to the book(s) you train your model on to ensure good results.

In order to get good results, you will likely have to train you model several times with different hyperparameters, adjusting the hyperparameters to reduce the model's error.

**Part 3 (10 pts):**
If you select books from project gutenberg written in a language other than English, you get these 10 points.
You'll need to provide translations for me of all the words used in Part 1.

**Part 4 (10 pts):**
Also train a [FastText model](https://radimrehurek.com/gensim/models/fasttext.html#gensim.models.fasttext.FastText) on your data.
This should be as simple as swapping out the `Word2Vec` line for a `FastText` line, but then you will have to potentially adjust additional hyperparameters for the FastText model.

Answer the question: which model performs better on your dataset and tests, and give a hypothesis as to why. (1 paragraph is enough.)

### Submission

Your final submission should include a 1 page write-up with each of the sections above that you completed, with your source code attached.
