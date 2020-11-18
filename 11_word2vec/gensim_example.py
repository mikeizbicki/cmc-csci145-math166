import gensim.downloader
from pprint import pprint

#model = 'word2vec-google-news-300'         # this one is trained on the most data
#model = 'fasttext-wiki-news-subwords-300'  # fasttext can create vectors for out of vocabulary words
#model = 'glove-twitter-25'                 # the smallest model, fits into memory easily
model = 'glove-wiki-gigaword-300'           # this is a reasonable compromise of size/accuracy

# load the model
vectors = gensim.downloader.load(model)

# similarities are the "opposite" of a distance: 
#   - low numbers far apart
#   - high numbers close
vectors.similarity('woman', 'man')

# get the most similar words;
# this is what you need for your assignment
vectors.most_similar('man')

# figure out which words don't belong in a set
vectors.doesnt_match(['man','woman','boy','girl','pig'])

vectors.doesnt_match(['clinton','biden','obama','trump'])

vectors.doesnt_match(['clinton','biden','obama','bush'])

# some example word analogies
# equation is: father - man + woman
vectors.most_similar(positive=['father','woman'],negative=['man'])

vectors.most_similar(positive=['father','daughter'],negative=['mother'])

vectors.most_similar(positive=['husband','woman'],negative=['man'])

vectors.most_similar(positive=['trump','woman'],negative=['man'])

vectors.most_similar(positive=['man','female'],negative=['male'])

# there's lots of analogies that don't work well
vectors.most_similar(positive=['donkey','republican'],negative=['democrat'])

vectors.most_similar(positive=['rooster','woman'],negative=['man'])

# is there gender bias in the vectors?
vectors.most_similar(positive=['baseball','woman'],negative=['man'])
vectors.most_similar(positive=['baseball'])

vectors.most_similar(positive=['basketball','woman'],negative=['man'])
vectors.most_similar(positive=['basketball'])
vectors.similarity('man', 'basketball')
vectors.similarity('woman', 'basketball')
vectors.similarity('man', 'sports')
vectors.similarity('woman', 'sports')

vectors.most_similar(positive=['engineer','woman'],negative=['man'])
vectors.most_similar(positive=['engineer'])
vectors.similarity('man', 'engineer')
vectors.similarity('woman', 'engineer')

vectors.most_similar(positive=['doctor','woman'],negative=['man'])
vectors.most_similar(positive=['doctor'])
vectors.similarity('man', 'doctor')
vectors.similarity('woman', 'doctor')
vectors.similarity('man', 'nurse')
vectors.similarity('woman', 'nurse')

vectors.most_similar(positive=['firefighter','woman'],negative=['man'])
vectors.most_similar(positive=['firefighter'])
vectors.similarity('man', 'firefighter')
vectors.similarity('woman', 'firefighter')

# Research questions that will dominate humanities/social science research over the next 10 years:
#
# 1. With a large set of analogies, we can measure the gender/racial/etc bias of word vectors.
#    This bias is a direct result of the training data used to train the vectors.
#    By training vectors on different corpora, we can measure the bias of those corpora.
#    This gives us a quantitative measure to determine whether a certain group is biased.
#
# 2. Domain specific test sets can be used to measure how good models perform in a specific domain.
#    Can we create good test sets and training procedures for IR/government domains?
#
#    For example:
#
#       CNN - democrat + republican = FoxNews
#
#       NYTimes - USA + Britain = BBC
#
#       Hezbollah - Lebanon + Palestine = Hamas
#
#       Trump - USA + China = Xi_Jinping
#
#       Republican - pro_life + democrat = pro_choice
#
#       Pelosi - democrat + republican = ?
#
#       Pelosi - California + Texas = ?
#
#       M16 - USA + Russia = AK47
#
#       Ohio_Class_Submarine - USA + China = Jin_Class_Submarine
#
