#!/usr/bin/python3

'''
This file calculates pagerank vectors for small-scale webgraphs.
See the README.md for example usage.
'''

import math
import torch
import gzip
import csv

import logging


class WebGraph():

    def __init__(self, filename, max_nnz=None, filter_ratio=None):
        '''
        Initializes the WebGraph from a file.
        The file should be a gzipped csv file.
        Each line contains two entries: the source and target corresponding to a single web link.
        This code assumes that the file is sorted on the source column.
        '''

        self.url_dict = {}
        indices = []

        from collections import defaultdict
        target_counts = defaultdict(lambda: 0)

        # loop through filename to extract the indices
        logging.debug('computing indices')
        with gzip.open(filename,newline='',mode='rt') as f:
            for i,row in enumerate(csv.DictReader(f)):
                if max_nnz is not None and i>max_nnz:
                    break
                import re
                regex = re.compile(r'.*((/$)|(/.*/)).*')
                if regex.match(row['source']) or regex.match(row['target']):
                    continue
                source = self._url_to_index(row['source'])
                target = self._url_to_index(row['target'])
                target_counts[target] += 1
                indices.append([source,target])

        # remove urls with too many in-links
        if filter_ratio is not None:
            new_indices = []
            for source,target in indices:
                if target_counts[target] < filter_ratio*len(self.url_dict):
                    new_indices.append([source,target])
            indices = new_indices

        # compute the values that correspond to the indices variable
        logging.debug('computing values')
        values = []
        last_source = indices[0][0]
        last_i = 0
        for i,(source,target) in enumerate(indices+[(None,None)]):
            if source==last_source:
                pass
            else:
                total_links = i-last_i
                values.extend([1/total_links]*total_links)
                last_source = source
                last_i = i

        # generate the sparse matrix
        i = torch.LongTensor(indices).t()
        v = torch.FloatTensor(values)
        n = len(self.url_dict)
        self.P = torch.sparse.FloatTensor(i, v, torch.Size([n,n]))
        self.index_dict = {v: k for k, v in self.url_dict.items()}
    

    def _url_to_index(self, url):
        '''
        given a url, returns the row/col index into the self.P matrix
        '''
        if url not in self.url_dict:
            self.url_dict[url] = len(self.url_dict)
        return self.url_dict[url]


    def _index_to_url(self, index):
        '''
        given a row/col index into the self.P matrix, returns the corresponding url
        '''
        return self.index_dict[index]


    def make_personalization_vector(self, query=None):
        '''
        If query is None, returns the vector of 1s.
        If query contains a string,
        then each url satisfying the query has the vector entry set to 1;
        all other entries are set to 0.
        '''
        n = self.P.shape[0]

        if query is None:
            v = torch.ones(n)

        else:
            v = torch.zeros(n)
            # FIXME: implement Task 2
        
        v_sum = torch.sum(v)
        assert(v_sum>0)
        v /= v_sum

        return v


    def power_method(self, v=None, x0=None, alpha=0.85, max_iterations=1000, epsilon=1e-6):
        '''
        This function implements the power method for computing the pagerank.

        The self.P variable stores the $P$ matrix.
        You will have to compute the $a$ vector and implement Equation 5.1 from "Deeper Inside Pagerank."
        '''
        with torch.no_grad():
            n = self.P.shape[0]

            # create variables if none given
            if v is None:
                v = torch.Tensor([1/n]*n)
                v = torch.unsqueeze(v,1)
            v /= torch.norm(v)

            if x0 is None:
                x0 = torch.Tensor([1/(math.sqrt(n))]*n)
                x0 = torch.unsqueeze(x0,1)
            x0 /= torch.norm(x0)

            # main loop
            xprev = x0
            x = xprev.detach().clone()
            for i in range(max_iterations):
                xprev = x.detach().clone()

                # compute the new x vector using Eq (5.1)
                # FIXME: Task 1
                # HINT: this can be done with a single call to the `torch.sparse.addmm` function,
                # but you'll have to read the code above to figure out what variables should get passed to that function
                # and what pre/post processing needs to be done to them

                # output debug information
                residual = torch.norm(x-xprev)
                logging.debug(f'i={i} residual={residual}')

                # early stop when sufficient accuracy reached
                if residual < epsilon:
                    break

            x = x0.squeeze()
            return x


    def search(self, pi, query='', max_results=10):
        '''
        Logs all urls that match the query.
        Results are displayed in sorted order according to the pagerank vector pi.
        '''
        n = self.P.shape[0]
        vals,indices = torch.topk(pi,n)

        matches = 0
        for i in range(n):
            if matches >= max_results:
                break
            index = indices[i].item()
            url = self._index_to_url(index)
            pagerank = vals[i].item()
            if url_satisfies_query(url,query):
                logging.info(f'rank={matches} pagerank={pagerank:0.4e} url={url}')
                matches += 1


def url_satisfies_query(url, query):
    '''
    This functions supports a moderately sophisticated syntax for searching urls for a query string.
    The function returns True if any word in the query string is present in the url.
    But, if a word is preceded by the negation sign `-`,
    then the function returns False if that word is present in the url,
    even if it would otherwise return True.

    >>> url_satisfies_query('www.lawfareblog.com/covid-19-speech', 'covid')
    True
    >>> url_satisfies_query('www.lawfareblog.com/covid-19-speech', 'coronavirus covid')
    True
    >>> url_satisfies_query('www.lawfareblog.com/covid-19-speech', 'coronavirus')
    False
    >>> url_satisfies_query('www.lawfareblog.com/covid-19-speech', 'covid -speech')
    False
    >>> url_satisfies_query('www.lawfareblog.com/covid-19-speech', 'covid -corona')
    True
    >>> url_satisfies_query('www.lawfareblog.com/covid-19-speech', '-speech')
    False
    >>> url_satisfies_query('www.lawfareblog.com/covid-19-speech', '-corona')
    True
    >>> url_satisfies_query('www.lawfareblog.com/covid-19-speech', '')
    True
    '''
    satisfies = False
    terms = query.split()

    num_terms=0
    for term in terms:
        if term[0] != '-':
            num_terms+=1
            if term in url:
                satisfies = True
    if num_terms==0:
        satisfies=True

    for term in terms:
        if term[0] == '-':
            if term[1:] in url:
                return False
    return satisfies


if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--personalization_vector_query')
    parser.add_argument('--search_query', default='')
    parser.add_argument('--filter_ratio', type=float, default=None)
    parser.add_argument('--alpha', type=float, default=0.85)
    parser.add_argument('--max_iterations', type=int, default=1000)
    parser.add_argument('--epsilon', type=float, default=1e-6)
    parser.add_argument('--max_results', type=int, default=10)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    g = WebGraph(args.data, filter_ratio=args.filter_ratio)
    v = g.make_personalization_vector(args.personalization_vector_query)
    pi = g.power_method(v, alpha=args.alpha, max_iterations=args.max_iterations, epsilon=args.epsilon)
    g.search(pi, query=args.search_query, max_results=args.max_results)
