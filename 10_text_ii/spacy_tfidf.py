import spacy
import wordfreq
import math
import pprint

# create the spacy doc
text = 'Apple is a rich company with lots of money and wealth.  Investing in Apple is a good investment.'
nlp = spacy.load("en_core_web_sm")
doc = nlp(text)

# create a dictionary mapping hashes to the number of occurances
tfs_hash = doc.count_by(spacy.attrs.IDS['LEMMA'])
#pprint.pprint(tfs_hash)

# compute the tf for words instead of hashes
tfs = { doc.vocab.strings[k]:counts_hash[k] for k in counts_hash.keys() if len(doc.vocab.strings[k])>1 }
#pprint.pprint(tfs)

# compute the idf for words
epsilon = 1e-20
idfs = { k:math.log (1/(epsilon+wordfreq.word_frequency(k, 'en', wordlist='best', minimum=0.0))) for k in tfs }
#pprint.pprint(idfs)

# compute the tf-idf scores
tfidfs = { k:tfs[k]*idfs[k] for k in tfs }
pprint.pprint(idfs)
