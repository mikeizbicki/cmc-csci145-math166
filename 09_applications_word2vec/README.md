# Homework: web search with word embeddings

**Overview:**

You will extend your pagerank project from the beginning of the semester to search for "similar" words using word embeddings.

This will help you see how the statistical theory we discussed for midterms 2-3 can be combined with the pagerank application of midterm 1.

> *NOTE:*
> I recommend using your solution to the original pagerank project as the basis for this project.
> If you would prefer not to for whatever reason, however, the `pagerank.py` file in this folder is my solution to the original pagerank project.

<!--
Task 1 is required, Tasks 2 and 3 are extra credit.
Each extra credit is worth up to 3% added to your final score in the class.
For example, if you have an 85% as your final grade,
but complete both extra credits,
your final percentage will be 91%.
-->

**Due date:**

Sunday, 11 December at midnight

**References:**

Word2Vec high-level overview: <https://blog.acolyer.org/2016/04/21/the-amazing-power-of-word-vectors/>

Word2Vec details with math: <http://arxiv.org/abs/1402.3722v1>

Interesting applications:

1. machine translation with word2vec: <https://arxiv.org/abs/1309.4168>

1. measuring gender bias in text corpora: <http://wordbias.umiacs.umd.edu/>

1. measuring racial bias in text corpora: <https://www.pnas.org/content/115/16/E3635/tab-figures-data>

1. measuring semantic shift in word meaning: <https://nlp.stanford.edu/projects/histwords>

## Task 1: finding similar words

<!--
#The file `pagerank.py` contains a completed solution to the pagerank problem.
-->
Recall that we can use the `pagerank.py` file to find the "highest quality" search results from the <https://lawfareblog.com> website.

For example, searching for "weapons" gives us:
```
$ python3 pagerank.py --data=./lawfareblog.csv.gz --search_query='weapons'
INFO:root:rank=0 pagerank=0.0011619674041867256 url=www.lawfareblog.com/slaughterbots-and-other-anticipated-autonomous-weapons-problems
INFO:root:rank=1 pagerank=0.0006675782497040927 url=www.lawfareblog.com/history-do-it-yourself-weapons-and-explosives-manuals-america
INFO:root:rank=2 pagerank=0.0006599930347874761 url=www.lawfareblog.com/lethal-autonomous-weapons-systems-first-and-second-un-gge-meetings
INFO:root:rank=3 pagerank=0.0006531275575980544 url=www.lawfareblog.com/lethal-autonomous-weapons-systems-recent-developments
INFO:root:rank=4 pagerank=0.0006407579057849944 url=www.lawfareblog.com/too-early-ban-us-and-uk-positions-lethal-autonomous-weapons-systems
INFO:root:rank=5 pagerank=0.0006338492385111749 url=www.lawfareblog.com/living-weapons-biological-warfare-and-international-security-gregory-koblentz
INFO:root:rank=6 pagerank=0.0005889176391065121 url=www.lawfareblog.com/critical-gaps-remain-defense-department-weapons-system-cybersecurity
INFO:root:rank=7 pagerank=0.0005780397332273424 url=www.lawfareblog.com/digital-strangelove-cyber-dangers-nuclear-weapons
INFO:root:rank=8 pagerank=0.0005346386460587382 url=www.lawfareblog.com/chemical-weapons-syria-enough-justify-use-force
INFO:root:rank=9 pagerank=0.0005290458793751895 url=www.lawfareblog.com/complexities-usg-covert-action-supply-weapons-syrian-rebels
```
and searching for "drones" gives us:
```
$ python3 pagerank.py --data=./lawfareblog.csv.gz --search_query='drones'
INFO:root:rank=0 pagerank=0.0006651429575867951 url=www.lawfareblog.com/future-violence-robots-and-germs-hackers-and-drones-confronting-new-age-threat
INFO:root:rank=1 pagerank=0.000632093520835042 url=www.lawfareblog.com/faa-wants-hear-you-about-privacy-and-domestic-drones
INFO:root:rank=2 pagerank=0.0006249933503568172 url=www.lawfareblog.com/ryan-calo-faas-setback-domestic-drones
INFO:root:rank=3 pagerank=0.0005648414953611791 url=www.lawfareblog.com/future-violence-now-hostile-use-drones
INFO:root:rank=4 pagerank=0.0005380361108109355 url=www.lawfareblog.com/lawfare-podcast-episode-5-missy-cummings-drones-drones-drones
INFO:root:rank=5 pagerank=0.0005302542704157531 url=www.lawfareblog.com/transcript-john-brennans-speech-yemen-and-drones
INFO:root:rank=6 pagerank=0.0005287894746288657 url=www.lawfareblog.com/no-more-drones-cia
INFO:root:rank=7 pagerank=0.0005287464591674507 url=www.lawfareblog.com/defending-drones-oxford-union
INFO:root:rank=8 pagerank=0.0005245042848400772 url=www.lawfareblog.com/readings-civilian-intelligence-agencies-and-use-armed-drones-ian-henderson
INFO:root:rank=9 pagerank=0.0005204146727919579 url=www.lawfareblog.com/wittes-v-oconnell-targeted-killing-and-drones
```
and searching for "targets" gives us:
```
$ python3 pagerank.py --data=./lawfareblog.csv.gz --search_query='targets'
INFO:root:rank=0 pagerank=0.0005385450785979629 url=www.lawfareblog.com/exclusive-nsa-program-can-target-thoughts-millions-targets-thousands-americans
INFO:root:rank=1 pagerank=0.0005327311810106039 url=www.lawfareblog.com/saudi-women-granted-new-rights-kurdistan-under-pressure-after-referendum-egypt-targets-dissidents
INFO:root:rank=2 pagerank=0.0005264155333861709 url=www.lawfareblog.com/turkeys-referendum-pushes-country-toward-authoritarianism-attack-targets-civilians-syrian-ceasefire
INFO:root:rank=3 pagerank=0.0005200763698667288 url=www.lawfareblog.com/british-anwar-al-awlaki-scenario-uk-targets-british-isil-member-syria-imminentcontinuous-threat
INFO:root:rank=4 pagerank=0.0005126169417053461 url=www.lawfareblog.com/middle-east-ticker-mosul-offensive-begins-us-fires-targets-yemen-and-what-intervention-syria-will
INFO:root:rank=5 pagerank=0.0005046890000812709 url=www.lawfareblog.com/us-forces-said-have-bombed-isis-targets-iraq
INFO:root:rank=6 pagerank=0.0005030750180594623 url=www.lawfareblog.com/trumps-big-middle-east-trip-us-targets-assad-forces-syria-and-iran-goes-polls
INFO:root:rank=7 pagerank=0.0005025876453146338 url=www.lawfareblog.com/international-fallout-trumps-immigration-and-refugee-order-us-raid-targets-aqap-yemen-greece-turkey
INFO:root:rank=8 pagerank=0.0005012631299905479 url=www.lawfareblog.com/der-spiegel-claims-germany-witholds-intel-militants-who-might-be-drone-strike-targets
INFO:root:rank=9 pagerank=0.0005012631299905479 url=www.lawfareblog.com/what-right-number-call-detail-records-42-targets-under-fisas-business-records-authority
```

Notice that the results for each of these terms do not overlap at all.

In a real search engine, however, we probably want to return results that mention "drones" or "targets" if someone searches for "weapons" since these terms are highly related.
Word vectors give us a tool for doing this.

In particular, we can use the [gensim](https://radimrehurek.com/gensim/) python library and pretrained word vectors to find words similar to the search word, and search for those as well.
For example, we can find words related to "weapons" by running the python code
```
>>> import gensim.downloader
>>> vectors = gensim.downloader.load('glove-twitter-25')
>>> vectors.most_similar('weapons')
[('drones', 0.8980589509010315),
 ('drone', 0.8965809345245361),
 ('assault', 0.8937929272651672),
 ('targets', 0.8834593296051025),
 ('firearms', 0.8833326697349548),
 ('weapon', 0.8730441927909851),
 ('hiv', 0.8625047206878662),
 ('laws', 0.8613511323928833),
 ('drug', 0.8555657863616943),
 ('concealed', 0.8548306822776794)
]
```
Notice that both the words "drones" and "targets" appear in this list of similar words.
The list isn't perfect though... "hiv" doesn't seem very similar to "weapons" to me.

Part of the problem is that I'm using a particularly poor model above that has a relatively large out-of-sample error.
The `glove-twitter-25` model is trained only on twitter data and has only 25 dimensions.
We've seen from our discussion about statistical learning theory that:

1. Increasing the number of training data points will improve generalization error, but won't improve training error.

1. Increasing the number of dimensions will improve training error but hurt generalization error.

In practice, there is lots of data available for training word embeddings in English.
State-of-the-art word embeddings are trained on multi-petabyte datasets of webpage crawls,
and so they can have a very high dimension and still achieve low generalization error.
For rarer languages, however, less training data is available,
and so models with fewer dimensions are more effective.

For this assignment, you won't have to train your own model from scratch,
and can instead use a pretrained model.
You must select a model different from `glove-twitter-25`, since that's a particularly bad one.
You won't be evaluated on how good of a model you select,
but I encourage you to spend some time experimenting with different models to see what works well for this application.
You can find a list of models built into gensim [here](https://github.com/RaRe-Technologies/gensim-data#models),
and there's many other open source models that people have released.

**Your Task:**

Modify the `pagerank.py` file so that it also searches for the keywords in the query and the 5 most similar words.
The results of your modified file when searching for "weapons" should look something like:
```
$ python3 pagerank.py --data=./lawfareblog.csv.gz --search_query='weapons'
INFO:root:rank=0 pagerank=0.004571518860757351 url=www.lawfareblog.com/why-did-you-wait-moral-emptiness-and-drone-strikes
INFO:root:rank=1 pagerank=0.0031107424292713404 url=www.lawfareblog.com/dc-district-court-dismisses-journalists-drone-lawsuit
INFO:root:rank=2 pagerank=0.0020231129601597786 url=www.lawfareblog.com/revived-cia-drone-strike-program-comments-new-policy
INFO:root:rank=3 pagerank=0.0019667143933475018 url=www.lawfareblog.com/us-court-appeals-dc-circuit-dismisses-suit-over-us-drone-strike
INFO:root:rank=4 pagerank=0.001178761012852192 url=www.lawfareblog.com/iran-shoots-down-us-drone-domestic-and-international-legal-implications
INFO:root:rank=5 pagerank=0.0011619674041867256 url=www.lawfareblog.com/slaughterbots-and-other-anticipated-autonomous-weapons-problems
INFO:root:rank=6 pagerank=0.0011276121949777007 url=www.lawfareblog.com/german-courts-weigh-legal-responsibility-us-drone-strikes
INFO:root:rank=7 pagerank=0.0008373793680220842 url=www.lawfareblog.com/shift-jsoc-drone-strikes-does-not-mean-cia-has-been-sidelined
INFO:root:rank=8 pagerank=0.0007856971933506429 url=www.lawfareblog.com/waiving-imminent-threat-test-cia-drone-strikes-pakistan
INFO:root:rank=9 pagerank=0.0007412837585434318 url=www.lawfareblog.com/drone-strike-errors-and-hostage-tragedy-mapping-issues-newly-catalyzed-debate
```

Notice that most of these articles do not mention the word "weapons",
but instead mention the word "drone".

> *NOTE:*
>
> Your numbers and order of results will likely be slightly different.
> That's okay as long as they're reasonable.
> For example, you should be returning some results that do not contain the search word,
> but contain related words.

> *HINT:*
>
> The easiest way to implement this task is to modify the `url_satisfies_query` function so that it calls gensim's `most_similar` function on each of the input query words,
> and returns `True` if any of those words are in the list.
> I recommend using only the 5 most similar words,
> as the words quickly become fairly unrelated.

## Task 2: ranking with word importance

There are several problems with the method above for including similar words in our search results.
For example:

1. Some words have lots of very similar words.
   Consider the most similar words for the search "biden":
   ```
   >>> vectors.most_similar('biden')
   [('hillary', 0.9419968128204346),
    ('christie', 0.9419272541999817), 
    ('romney', 0.9377604722976685), 
    ('potus', 0.9371991157531738), 
    ('reid', 0.9328151345252991), 
    ('boehner', 0.9311597347259521), 
    ('clinton', 0.9253027439117432), 
    ('clegg', 0.9072402119636536), 
    ('miliband', 0.9010352492332458), 
    ('zimmerman', 0.8950174450874329)
    ]
   ```
   All of the results above are more similar to "biden" than any word is to "weapons".
   So if we are only looking at the top5 words for everything,
   then some words will get much better matches than others.
   It would be better to have an adaptive method that can automatically determine the top number of query results to use.

1. If someone is searching for "weapons",
   we should probably weight articles about weapons higher than articles about "drones",
   and we should rank articles about both "drones" and "weapons" the highest.
   Our previous method ranks all of these results equally.

One simple method for fixing both of these problems to define a `query_score` for each webpage in addition to the pagerank score `pi`.
This `query_score` would be a real number that depends on both the query and the article's text,
and would be high whenever the query and text are related to each other.
Then the webpage's ranking would be defined by:
```
ranking = pi * query_score
```
Instead of only by the pagerank `pi`.

There are many possible ways to define the query score.
For example, if we define the `query_score` to be 0 when `url_satisfies_query` returns `False` and 1 when it returns `True`,
then this algorithm reduces to the previous algorithm.

A better method is to use the similarity scores using the following pseudocode:
```
score = 0
let S be the set of words similar to the query string
for each word in S:
    let n be the number of times word appears in document
    let word_similarity be the similarity score of word
    score += n * word_similarity**p
```
By adjusting the p hyperparameter, we can control how important the word similarities must be.
A typical `p` value would be between 30-60, since this will result in scores roughly on a similar scale as the pagerank vectors.
You should hard-code this value to something that you think gives reasonable results.

> *NOTE:*
> 
> The optimal value for `p` will depend on the particular word embeddings you select.
> In real world search engines, the optimal value would be learned from the data.
> We would set up a classification problem where the output variable `Y` to be predicted is whether or not a user clicks on one of the links provided,
> and the input variable `X` would be the search term.
> The hypothesis class would be the pseudo-code you've written above with `p` as the only parameter to be learned.
> Then you can use gradient descent to solve for `p`.
>
> You won't have to do this.
> It is hard to implement because this is a non-standard hypothesis class not found in libraries scikit-learn,
> and acquiring reasonable training data is difficult for companies not already operating large-scale search engines.
>
> Google uses classification problems like this internally all over their search engine.
> When Google was first founded, it was common to hear people saying "Google uses machine learning the way Microsoft uses the if statement."
> Now all the major tech companies have machine learning problems like this internally that they are solving,
> so people don't say this anymore.

**Task:**

Implement the "better method" described above.
You will have to modify the `WebGraph.search` function so that it implements the pseudocode described above,
and orders documents according to the `ranking` score instead of according to `pi`.

<!--
## Task 3 (Extra Credit): Custom word vectors

In this task, you will train your own word vectors using gensim.
Training quality word vectors requires a lot of data,
and the lawfareblog dataset does not contain sufficient data by itself.
Instead, I recommend training on all of wikipedia.
You can find instructions for doing that with gensim at:
<https://textminingonline.com/training-word2vec-model-on-english-wikipedia-by-gensim>

Following these instructions takes about 24-48 hours of compute time on a standard laptop,
and about 20Gb of disk space.

**Task:**
Train your own word vectors, and demonstrate that they work reasonably well by performing analogy evaluations on them.

**Submission:**
Schedule a zoom meeting with me to demonstrate that your trained word vectors work.
-->

## Submission

Modify your pagerank github as follows:

1. Add the new `pagerank.py` file.

1. Update the `README.md` file from your pagerank homework assignment so that the output of the commands run in the "Submission" section contains the output from your newly modified `pagerank.py` file instead of the original.

Submit to sakai a link to your github repo.

<!--
## Extra Credit: Multilingual Word Embeddings

Facebook has released a set of multilingual word 
-->
