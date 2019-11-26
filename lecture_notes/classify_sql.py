#!/usr/bin/env python3

# process command line args
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--db',default='scrapedata.db')
parser.add_argument('--domains',default=['thegoldenantlers.com','scrippsvoice.com','cmcforum.com','claremontindependent.com','tsl.news'],nargs='+')
parser.add_argument('--ngrams',type=int,default=1)
parser.add_argument('--limit')
parser.add_argument('--num_words',type=int,default=20)
parser.add_argument('--num_eig',type=int,default=10)
parser.add_argument('--penalty',type=str,default='l2')
parser.add_argument('--C',type=float,default=1.0)
parser.add_argument('--no_biggest_words',action='store_true')
parser.add_argument('--norm',type=int,default=2)
parser.add_argument('--no_cocluster',action='store_true')
parser.add_argument('--features',choices=['counts','tf','tfidf'],default='tfidf')
args = parser.parse_args()

# generic imports
import datetime
import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from sklearn.cluster import SpectralCoclustering
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV

def plot_matrix(mat,filename,force_no_cocluster=False):
    print(datetime.datetime.now(),'plot_matrix')
    print('  mat.shape=',mat.shape)
    plt.figure(figsize=(10,4))

    # set the x-axis to only include the biggest words
    if not args.no_biggest_words:
        l2_norms=np.linalg.norm(mat,axis=0,ord=args.norm)
        indices = l2_norms.argsort()[-args.num_words:]
        mat = mat[:,indices]
        words = [ all_feature_names[i] for i in indices ]
        plt.xticks(ticks=range(0,len(words)),labels=words,rotation=-90)

    # cocluster the axes
    if not args.no_cocluster and not force_no_cocluster:
        clustering = SpectralCoclustering(n_clusters=6,random_state=1).fit(mat)
        col_indices = np.argsort(clustering.column_labels_)
        mat = mat[:,col_indices]
        try:
            words = [ words[i] for i in col_indices ]
            plt.xticks(ticks=range(0,len(words)),labels=words,rotation=-90)
        except:
            pass

    # plot the figure
    plt.imshow(
            mat,
            aspect='auto',
            cmap='RdBu',
            norm=colors.SymLogNorm(linthresh=0.03, linscale=0.03, vmin=-1e6, vmax=1e6)
            );

    if mat.shape[0]==5:
        plt.yticks(ticks=[0,1,2,3,4],labels=model.classes_)
        plt.ylim(-0.5,4.5)

    plt.colorbar()
    plt.tight_layout()
    plt.savefig(filename)

# prepare sqlite3 connection
print(datetime.datetime.now(),'connecting to database')
import sqlite3
connection = sqlite3.connect(args.db)
cursor = connection.cursor()

print(datetime.datetime.now(),'query db')
where_clause='and ('+' or '.join([f"d='{domain}'" for domain in args.domains])+')'

if args.limit is None:
    limit_clause=''
else:
    limit_clause=f'limit {args.limit}'

sql=f'''
select
    min(id) as id,
    body,
    title,
    replace(domain,'www.','') as d,
    strftime("%Y",pub_time) as year
from articles
where
    (year is not null or (d like '%scripps%' and length(body)>1000))
    {where_clause}
group by d,title
order by id asc
{limit_clause}
;
'''
cursor.execute(sql)
rows = cursor.fetchall()

print(datetime.datetime.now(),'extracting data from query result')
text = [ row[2] for row in rows ]
labels = [ row[3] for row in rows ]
print('  set(labels)=',set(labels))

print(datetime.datetime.now(),'CountVectorizer()')
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer(ngram_range=(1,args.ngrams),stop_words='english')
features = count_vect.fit_transform(text)
features = features.astype(np.float64)
all_feature_names = count_vect.get_feature_names()
print('  features.shape=',features.shape)
print('  features.dtype=',features.dtype)

if args.features=='tf':
    print(datetime.datetime.now(),'TF')
    from sklearn.feature_extraction.text import TfidfTransformer
    tf_transformer = TfidfTransformer(use_idf=False).fit(features)
    features = tf_transformer.transform(features)
    print('  features.shape=',features.shape)

if args.features=='tfidf':
    print(datetime.datetime.now(),'TF-IDF')
    from sklearn.feature_extraction.text import TfidfTransformer
    tf_transformer = TfidfTransformer(use_idf=True).fit(features)
    features = tf_transformer.transform(features)
    print('  features.shape=',features.shape)

print(datetime.datetime.now(),'PCA')
if args.num_eig>0:
    gram = np.dot(np.transpose(features),features)
    print('  gram.shape=',gram.shape)
    w, v = scipy.sparse.linalg.eigsh(gram,k=args.num_eig)
    print('  w.shape=',w.shape)
    print('  v.shape=',v.shape)

    plt.figure(figsize=(20,10))
    plt.bar(range(0,args.num_eig),w)
    plt.savefig('img/mat/eigenvalues.png')

    plot_matrix(np.transpose(v),filename='img/mat/eigenvectors.png') #,force_no_cocluster=True)

print(datetime.datetime.now(),'logreg')
model = LogisticRegression(
        penalty=args.penalty,
        C=args.C,
        solver='liblinear',
        class_weight='balanced',
        multi_class='auto'
        )
model.fit(features, labels)
print('  model.coef_.shape=',model.coef_.shape)

plot_matrix(model.coef_,f'img/mat/coefs_{args.penalty}_{args.C}.png')
