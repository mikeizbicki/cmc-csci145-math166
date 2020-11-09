import spacy
import unicodedata

def feature_hash(token, d=10, seed=0, prime=19134702400093278081449423917):
    '''
    this function uses a linear congruential generator (LCG) to add randomness to the hash values;
    this let's us adjust which values will have collisions by tuning the seed;
    see: https://en.wikipedia.org/wiki/Linear_congruential_generator
    '''
    random_hash = (token.lemma+seed)*prime
    return random_hash%d

import spacy.lang.en
en = spacy.lang.en.English()
doc = en('Apple is looking at buying U.K. startup for $1 billion, and San Francisco is considering banning sidewalk delivery robots, and The Killers is an awesome band.')
for token in doc:
    # stop word removal
    if not token.is_stop:

        # don't consider punctuation
        if not (len(token.lemma_)==1 and unicodedata.category(token.lemma_).startswith('P')):

            print('  ',token.lemma_, feature_hash(token))



