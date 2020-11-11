import spacy
import torch

nlp = spacy.load('en_core_web_sm',  disable=["parser", "ner"])

def make_features_batch(xs, ngrams, d, prime=19134702400093278081449423917, seed=0):
    '''
    computes the features of the string x using ngrams in dimension d
    '''

    indexes = []
    values = []

    for i,x in enumerate(xs):

        # compute the lemmas
        doc = nlp(x)
        lemmas = []
        for token in doc:
            #if not token.is_stop:
            lemmas.append((token.lemma+seed)*prime%d)
        lemmas.sort()

        # compute the lemmas
        doc = nlp(x)
        lemmas = []
        grams = []
        tokens = list(doc)
        for n in range(1,ngrams+1):
            for j in range(len(tokens)-n+1):
                gram = tokens[j:j+n]
                grams.append(' '.join([ token.lemma_ for token in gram]))
                lemma_sum = sum([ token.lemma for token in gram])
                lemmas.append((lemma_sum+seed)*prime%d)
        lemmas.sort()

        # compute indexes and values for x
        indexes_x = []
        values_x = []
        if len(lemmas) > 0:
            indexes_x.append(lemmas[0])
            values_x.append(1)
        for lemma in lemmas[1:]:
            if lemma==lemmas[-1]:
                values_x[-1] += 1
            else:
                indexes_x.append(lemma)
                values_x.append(1)

        # add to the batch
        indexes.extend([[i,index] for index in indexes_x])
        values.extend(values_x)

    return torch.sparse.FloatTensor(
            torch.LongTensor(indexes).t(),
            torch.FloatTensor(values),
            torch.Size([len(xs),d])
            )

def make_features(x, ngrams, d, prime=19134702400093278081449423917, seed=0, debug=False):
    '''
    computes the features of the string x using ngrams in dimension d
    '''
    doc = nlp(x)

    # compute the lemmas
    lemmas = []
    grams = []
    tokens = list(doc)
    for n in range(1,ngrams+1):
        for j in range(len(doc)-n+1):
            gram = tokens[j:j+n]
            grams.append(' '.join([ token.lemma_ for token in gram]))
            lemma_sum = sum([ token.lemma for token in gram])
            lemmas.append((lemma_sum+seed)*prime%d)
    lemmas.sort()

    # print debug information
    if debug:
        print('n-grams:')
        for gram,lemma in zip(grams,lemmas):
            print(' hash_value='+str(lemma)+'   n-gram='+str(gram))

    # construct the sparse tensor 
    indexes = []
    values = []
    if len(lemmas) > 0:
        indexes.append(lemmas[0])
        values.append(1)
    for lemma in lemmas[1:]:
        if lemma==lemmas[-1]:
            values[-1] += 1
        else:
            indexes.append(lemma)
            values.append(1)

    indexes = [[0,index] for index in indexes]
    return torch.sparse.FloatTensor(
            torch.LongTensor(indexes).t(),
            torch.FloatTensor(values),
            torch.Size([1,d])
            )

v = make_features('this is a test',2,12)
print("v.shape=",v.shape)
print("v=",v)


x = torch.ones([3,4])
def nnz(x):
    return (x!=0).sum()
print("nnz(x)=",nnz(x))
