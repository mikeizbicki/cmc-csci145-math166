# CSCI145 / MATH166: Data Mining
<!--
FIXME:
Include these links in the right topic folders:

1. Why I don't like notebooks https://www.youtube.com/watch?v=7jiPeIFXb6U

1. NeurIPS keynote on software engineering in data mining: https://nips.cc/virtual/2020/public/invited_16166.html

1 Tutorial on fairness: https://www.youtube.com/watch?v=jIXIuYdnyyk

1. AUC/ROC curves https://www.youtube.com/watch?v=4jRBRDbJemM

1. A cool fasttext application: https://mixedname.com/english_klingon_feminine_names

1. cross-lingual embeddings: https://ruder.io/cross-lingual-embeddings/
-->

<center>
<img width='400px' src='machine_learning_2x.png' />
</center>

Important links:

1. [Data Mining vs Machine Learning vs Artificial Intelligence vs Statistics](https://stats.stackexchange.com/questions/5026/what-is-the-difference-between-data-mining-statistics-machine-learning-and-ai)
1. [What do data scientists get paid?](https://www.levels.fyi/comp.html?track=Data%20Scientist)

## About the Instructor

|||
|-|-|
| Name | Mike Izbicki (call me Mike) |
| Email | mizbicki@cmc.edu |
| Office | Adams 216 |
| Office Hours | See [Issue #69](https://github.com/mikeizbicki/cmc-csci145-math166/issues/69) |
| Zoom | See [Issue #70](https://github.com/mikeizbicki/cmc-csci145-math166/issues/70) |
| Webpage | https://izbicki.me |
| Research | Machine Learning (see [izbicki.me/research.html](https://izbicki.me/research.html) for some past projects) |

Fun facts:
1. grew up in San Clemente (~1 hr south of Claremont)
1. 7 years in the navy
    1. nuclear submarine officer, personally converted >10g of uranium into pure energy
    1. worked at National Security Agency (NSA)
    1. left Navy as a [conscientious objector](https://www.nytimes.com/2011/02/23/nyregion/23objector.html)
1. phd/postdoc at UC Riverside
1. taught in [DPRK (i.e. North Korea)](https://pust.co)

## About the Course

**General Information:**

1. This is the theory course for CMC's Data Science major
1. Prepare you for industry or graduate school
    1. Especially for machine learning technical interviews
    1. No SQL in this course => that's CSCI133 Big Data

**Learning Objectives:**

1. See the [Jupyter notebook](intro.ipynb)

1. Exposure to *research-level* data mining
    1. Understand the latest algorithms...
       but algorithms get outdated fast.

    1. The real goal is to teach you how to read research-level papers and math so that you can understand future techniques by yourself

1. Major concepts
    1. Techniques
        1. Eigen-methods for data mining
        1. Logistic regression
        1. Kernel methods
        1. Neural networks
        1. word2vec
        1. Small amount of deep learning (transformers, CNNs, etc.)
    1. Math
        1. Bias/variance trade-off
        1. VC Dimension theorem (fundamental theorem of statistical learning)
        1. Regularization (L1, L2, elastic net, weight decay, early stopping, etc.)
        1. Optimization algorithms (gradient descent, stochastic gradient descent, ADAM, etc.)
    1. Programming:
        1. Writing code that is easy to deploy
    1. Focus on text/web/social media examples

1. Ethical implications of data mining
   
   Pet peeve: You can't fully understand the ethics if you don't understand the technical details

1. Apply data mining libraries (PyTorch, scikit-learn, GenSim, spaCy, etc.) 
    1. Teaching you how to use these libraries is NOT the primary goal of the course
    1. In-person class time will focus on the math, and I'm expecting you can figure out how to use the libraries on your own

**Prerequisite knowledge:**

<!-- FIXME -->
1. linear algebra
    1. eigenvectors
1. computation
    1. big-o analysis
    1. git
    1. download/use python libraries
1. statistics
    1. super basic probability
    1. exposure to linear/logistic regression helpful but not required

**Textbook:**

I will provide all the reference material for this class.
You don't have to buy anything.

1. *Learning from Data* by Yaser S. Abu-Mostafa, Malik Magdon-Ismail, and Hsuan-Tien Lin

    I am providing you all a free copy.
    It is yours to keep forever if you'd like (or you can return it to me at the end of the semester and I'll pass it on to future students).
    Feel free to highlight/take notes/etc in it as if it were your own book, because it is.

1. *Understanding Machine Learning: From Theory to Algorithms* by Shai Shalev-Shwartz and Shai Ben-David

    Freely available [from Shalev-Shwartz's website](https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/)

1. Lots of research papers / lecture notes

<!--
1. Christopher Bishop's *Pattern Recognition and Machine Learning*.  [Download a free pdf copy from Microsoft Research.](https://www.microsoft.com/en-us/research/people/cmbishop/prml-book/)
1. Peterson's *The Matrix Cookbook* is a handy reference for multivariable calculus, and [you can download a free copy here](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf).
1. Other articles as posted in the schedule below.
1. Other popular (and freely available) data mining books include [Data Mining: Concepts and Techniques](http://myweb.sabanciuniv.edu/rdehkharghani/files/2016/02/The-Morgan-Kaufmann-Series-in-Data-Management-Systems-Jiawei-Han-Micheline-Kamber-Jian-Pei-Data-Mining.-Concepts-and-Techniques-3rd-Edition-Morgan-Kaufmann-2011.pdf) by Jiawei Han, Micheline Kamber and Jian Pei and [Data Mining](http://www.charuaggarwal.net/Data-Mining.htm) by Charu C. Aggarwal.  We probably won't use these books at all though.
-->

**Grades:**

| Category                          | Percent   | Approximate Date    |
| --------------------------------- | --------- | ------------------- |
| Projects                          | 30        | Every 2-3 weeks     |
| Quizzes                           | 0         |                     |
| Midterm 1 (Pagerank)              | 15        | Week 03             |
| Midterm 2 (Learning from Data)    | 15        | Week 08             |
| Midterm 3 (Text mining)           | 15        | Week 13             |
| Final                             | 25        |                     |

Projects:

1. 4-7 projects
1. All of them must be completed on the lambda server (i.e. using ssh+bash+vim)
   
   Lambda server has 80 CPUs + 8 GPUs
1. I'm expecting almost everyone will get full credit, and these will act as a "grade boost"

<!--
| Project: Search Engine I          | 5         | Week 2              |
| Project: Statistical Learning I   | 5         |                     |
| Project: Statistical Learning II  | 5         |                     |
| Project: Transfer Learning        |           |                     | https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
| Project: Twitter                  | 5         |                     |
| Project: Search Engine II         | 5         |                     |
-->

Quizzes:

1. There will be 1 quiz per midterm testing definition memorization.
1. I will give you the quiz before you take it.
1. They are not worth any points,
    but **you must get 100% on the quiz or you will fail the class**.
1. Unlimited retakes, but each retake results in a -1% off your final grade.

Midterms:

1. No programming, only math
1. Take home, unlimited time, open note
1. Very hard exams. (Historically, average in the 70s.  No curve.)

Final:

1. Oral exam
1. The purpose is to help prepare you for interviews.
1. The last week of class will be dedicated to prep.
1. The final grade can replace your lowest midterm grade, if that would improve your overall grade in the class.

This is a **hard** class.

1. The material is intrinsically hard
    1. Very few people find linear algebra, statistics and programming to ALL be easy subjects, and this class combines them all
    1. There's a reason people who understand this material get paid big salaries at FAANG

1. You will have to read the required references.

    Not all the material will be covered in lectures,
    and that's intentional to force you to get practice reading research-level data mining text.

1. Comments from previous students:

    1.  > Holy fucking shit this was a hard class.
        > I had no idea there was so much god damned fucking math involved in a CS class.
        > You should warn students about that.

    1.  > I spent 20+ hours per week on this class, and still only got a B.
        > The class is too hard and you should make it easier.

    Unfortunately, I can't remove the math from this class, and I can't make the class easier.
    Otherwise, you wouldn't be learning the material needed to pass a technical interview / get a good job / go to grad school.

    <img src=math.webp />

> **NOTE:**
> In all of my other courses, I include required reading/watching tasks to learn about CS/DS culture.
> This course doesn't have these tasks because there is already a LOT of textbook reading that you will have to complete.

**Late Work Policy:**

You lose 20% on projects for each day late.
It is still typically better to submit a correct assignment late than an incorrect one on time.

If you collaborate with other students, 
you get an automatic 2 day extension on any project.

**Collaboration Policy:**

You are encouraged to discuss all labs and projects with other students,
subject to the following constraints:

1. you must be the person typing in all code for your assignments, and
1. you must not copy another student's code.

You may use any online resources you like as references.

Basically, I'm trusting you all to be adults.
You are ultimately responsible for ensuring you learn the material!
So do what will help you learn best.

> **WARNING:**
> All material in this class is cumulative.
> If you work "too closely" with another student on an assignment,
> you won't understand how to complete subsequent assignments,
> and you will quickly fall behind.
> You should view collaboration as a way to improve your understanding,
> not as a way to do less work.

<!--
## Schedule

| Homework              | Topic                                         |
| --------------------- | --------------------------------------------- |
| 1                     | Review                                        |
| 2                     | Pagerank: math                                |
| 3                     | Pagerank: implementation                      |
| 4                     | SLT: math                                     |
| 5                     | SLT: implementation (theoretical)             |
| 6                     | SLT: implementation (practical)               |
| 7                     | SGD: math                                     |
| 8                     | SGD: implementation                           |
| 9                     | Word2Vec                                      |
| 11                    | Sentiment Analysis                            |

| Week | Date        | Topic                                                                |
| ---- | ----------- | -------------------------------------------------------------------- |
| 1    | Mon, Aug 24 | Course intro                                                         |
| 1    | Wed, Aug 26 | Computational Linear Algebra                                         |
| 2    | Mon, Aug 31 | Pagerank                                                             |
| 2    | Wed, Sep 2  | Pagerank                                                             |
| 3    | Mon, Sep 7  | Statistical Learning Theory                                          |
| 3    | Wed, Sep 9  | Statistical Learning Theory                                          |
| 4    | Mon, Sep 14 | Statistical Learning Theory                                          |
| 4    | Wed, Sep 16 | Statistical Learning Theory                                          |
| 5    | Mon, Sep 21 | Logistic Regression                                                  |
| 5    | Wed, Sep 23 | Logistic Regression                                                  |
| 6    | Mon, Sep 28 | Kernels / neural networks / k-nearest neighbor / decision trees      |
| 6    | Wed, Sep 30 | Kernels / neural networks / k-nearest neighbor / decision trees      |
| 7    | Mon, Oct 5  | Stochastic gradient descent                                          |
| 7    | Wed, Oct 7  | Stochastic gradient descent                                          |
| 8    | Mon, Oct 12 | Regularization                                                       |
| 8    | Wed, Oct 14 | Regularization                                                       |
| 9    | Mon, Oct 19 | Hashing trick / random projections                                   |
| 9    | Wed, Oct 21 | Hashing trick / random projections                                   |
| 10   | Mon, Oct 26 | Word2Vec                                                             |
| 10   | Wed, Oct 28 | Word2Vec                                                             |
| 11   | Mon, Nov 2  | Word2Vec: FastText                                                   |
| 11   | Wed, Nov 4  | Word2Vec: translation  |
| 12   | Mon, Nov 9  | Word2Vec: bias                                                       |
| 12   | Wed, Nov 11 | Word2Vec: history                                                    |
| 13   | Mon, Nov 16 | Other Applications                                                   |
| 13   | Wed, Nov 18 | Other Applications                                                   |
| 14   | Mon, Nov 23 | Other Applications                                                   |
-->

<!--
| Week | Date | Topic | Assignment |
| ---- | --- | --- | --- |
| 1  | Tues, 3 Sept  | <p>Introduction</p> Examples:<ol><li>[/r/dataisbeautiful](https://www.reddit.com/r/dataisbeautiful/top/)<li>okcupid [pictures](https://theblog.okcupid.com/dont-be-ugly-by-accident-b378f261dea4), [messages](https://theblog.okcupid.com/exactly-what-to-say-in-a-first-message-2bf680806c72), and [lies](https://theblog.okcupid.com/the-big-lies-people-tell-in-online-dating-a9e3990d6ae2)<li>[NLP analysis of net neutrality comments](https://medium.com/hackernoon/more-than-a-million-pro-repeal-net-neutrality-comments-were-likely-faked-e9f0e3ed36a6)<li>[sattelite images of cars affect stock prices](https://theoutline.com/post/1169/jc-penney-satellite-imaging?zd=1&zi=qmayberw)<li>[digitalnk: gender and North Korean posters](https://digitalnk.com/blog/2017/09/30/gender-distribution-in-north-korean-posters/); [posters](https://www.businessinsider.com/kim-jong-il-kim-jong-un-north-korea-propoganda-2011-12)</ol>Ethics:<ol><li>[Target and pregnancy](https://www.nytimes.com/2012/02/19/magazine/shopping-habits.html?pagewanted=1&_r=1&hp)<li>[Data mining algorithms determine prison time](https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing)<li>[Google Gorilla mistake](https://twitter.com/jackyalcine/status/615329515909156865)</ol> | HW 0 |
| 1  | Thur, 5 Sept  | Bishop 1.2: Probability Theory Review | |
| 2  | Tues, 10 Sept | Bishop 1.3-1.6: Introduction to Machine Learning | |
| 2  | Thur, 12 Sept | Bishop 1.3-1.6: Introduction to Machine Learning | Project (proposal)<br/>HW 1 |
| 3  | Tues, 17 Sept | **NO CLASS** (Mike at [ECML-PKDD](http://ecmlpkdd2019.org/)) | |
| 3  | Thur, 19 Sept | **NO CLASS** (Mike at [ECML-PKDD](http://ecmlpkdd2019.org/)) |  |
| 4  | Tues, 24 Sept | Bishop 3.1: Regression: Linear Regression | |
| 4  | Thur, 26 Sept | Bishop 3.2: Regression: The Bias-Variance Trade-off | Quiz 1 |
| 5  | Tues, 1 Oct   | Bishop 4.1: Classification: Discriminant Functions | |
| 5  | Thur, 3 Oct   | Bishop 4.2: Classification: Generative Models | |
| 6  | Tues, 8 Oct   | Bishop 4.3: Classification: Discriminative Models | HW 2 |
| 6  | Thur, 10 Oct  | [Leon Bottou's SGD paper](https://datajobs.com/data-science-repo/Stochastic-Gradient-Descent-[Leon-Bottou].pdf) | |
| 7  | Tues, 15 Oct  | Text Processing: [bag of words](https://en.wikipedia.org/wiki/Bag-of-words_model#Example_usage:_spam_filtering), [tf-idf](https://skymind.ai/wiki/bagofwords-tf-idf), [n-grams](https://en.wikipedia.org/wiki/N-gram),[hashing trick](https://booking.ai/dont-be-tricked-by-the-hashing-trick-192a6aae3087), [zipf's law](https://en.wikipedia.org/wiki/Zipf%27s_law) <br/>Python NLP libraries: [TextBlob](https://textblob.readthedocs.io/en/dev/), [spacy](https://spacy.io/), [neuralcoref](https://github.com/huggingface/neuralcoref), [NLTK](https://www.nltk.org/), [textstat](https://pypi.org/project/textstat/) | Quiz 2<br/>Project (checkup) |
| 7  | Thur, 17 Oct  | Text Processing: word2vec [high level overview](https://blog.acolyer.org/2016/04/21/the-amazing-power-of-word-vectors/), [details with math](http://arxiv.org/pdf/1402.3722v1.pdf) | |
| 8  | Tues, 22 Oct  | **NO CLASS** (Fall Break) | |
| 8  | Thur, 24 Oct  | Text Processing: more word2vec | |
| 9  | Tues, 29 Oct  | Text Processing: [translating with word2vec](https://arxiv.org/abs/1309.4168), [doc2vec](https://arxiv.org/abs/1405.4053), [fastText](https://fasttext.cc) | HW 3 |
| 9  | Thur, 31 Oct  | more word2vec examples: [gender bias](http://wordbias.umiacs.umd.edu/), [racial bias](https://www.pnas.org/content/115/16/E3635/tab-figures-data), [histwords](https://nlp.stanford.edu/projects/histwords/), [temporal word analogies](https://www.aclweb.org/anthology/P17-2071/) [political words](https://arxiv.org/abs/1711.05603)<br/>exploiting Twitter metadata: [trump tweets](http://varianceexplained.org/r/trump-tweets/) |  |
| 10 | Tues, 5 Nov   | **NO CLASS** (Mike at [CIKM 2019](http://www.cikm2019.net/)) | |
| 10 | Thur, 7 Nov   | **NO CLASS** (Mike at [CIKM 2019](http://www.cikm2019.net/)) |  |
| 11 | Tues, 12 Nov  | Linear algebra review | |
| 11 | Thur, 14 Nov  | Bishop 12.1: The multivariate gaussian / PCA<br>[Lecture Notes from Berkeley](https://people.eecs.berkeley.edu/~jrs/189/lec/08.pdf) | ~~HW 4~~ |
| 12 | Tues, 19 Nov  | Bishop 12.1: The multivariate gaussian / PCA | |
| 12 | Thur, 21 Nov  | Bishop 12.1: The multivariate gaussian / PCA<br/><a href='https://colah.github.io/posts/2014-10-Visualizing-MNIST/'>Colah's Visualizing MNIST</a> | |
| 13 | Tues, 26 Nov  | Bishop 9.2: Mixtures of Gaussians / k-means clustering<br/>Scikit learn <a href='https://scikit-learn.org/stable/auto_examples/linear_model/plot_sparse_logistic_regression_mnist.html'>1</a>, <a href='https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_assumptions.html#sphx-glr-auto-examples-cluster-plot-kmeans-assumptions-py'>2</a>, <a href='https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_covariances.html#sphx-glr-auto-examples-mixture-plot-gmm-covariances-py'>3</a>, <a href='https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm.html#sphx-glr-auto-examples-mixture-plot-gmm-py'>4</a>, <a href='https://www.naftaliharris.com/blog/visualizing-k-means-clustering/'>visualizing k-means</a> | ~~Quiz 3~~ |
| 13 | Thur, 28 Nov  | **NO CLASS** (Thanksgiving) |  |
| 14 | Tues, 3 Dec   | Bishop 7.1: SVMs<br/>[youtube](https://www.youtube.com/watch?v=3liCbRZPrZA), [youtube](https://www.youtube.com/watch?v=ndNE8he7Nnk), [scikit-learn](https://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html) | HW 5 |
| 14 | Thur, 5 Dec   | Bishop 6.1,6.2: Kernel methods | |
| 15 | Tues, 10 Dec  | Bishop 5.1: Neural networks | |
| 15 | Thur, 12 Dec  | Project Presentations | |
-->

<!--
Bishop 5.1: Feed Forward Neural Networks | HW 4 |
Bishop 6.1,6.2: Kernel Methods | Quiz 3 |
Bishop 6.4.1-6.4.3: Gaussian Processes |  |
Bishop 7.1: Support Vector Machines | |
Bishop 9.1: K-Means Clustering | HW 5 |
**NO CLASS** (Thanksgiving) |  |
Bishop 9.2: Mixtures of Gaussians |  |
Hierarchical clustering | |
Bishop 12.1: Principle Component Analysis |  |
The hashing trick (revisited) | Quiz 4 |
-->

<!--
Possible topics:
1. Leon Bottou's large scale learning with stochastic gradient descent / stochastic gradient descent tricks
1. Fully understanding the hashing trick
1. Feature hashing for large scale multitask learning
1. Random Projections and the Johnson-Lindenstrauss Lemma
-->

<!--
## Ethics

* Microsoft Tay

* Target: https://www.forbes.com/sites/kashmirhill/2012/02/16/how-target-figured-out-a-teen-girl-was-pregnant-before-her-father-did/

* Crime recidivism: https://advances.sciencemag.org/content/4/1/eaao5580.full https://www.heinz.cmu.edu/media/2017/january/automate-fairness-machine-learning-discrimination https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing

* Self driving cars: https://www.nytimes.com/2018/03/19/technology/uber-driverless-fatality.html

* Images: https://www.theverge.com/2018/1/12/16882408/google-racist-gorillas-photo-recognition-algorithm-ai

* https://freedom-to-tinker.com/2019/08/23/deconstructing-googles-excuses-on-tracking-protection/

* Border security:
https://www.reddit.com/r/legaladvice/comments/cyr3g3/i_am_an_american_citizen_yesterday_at_lax_i_was/
https://www.reddit.com/r/privacy/comments/cwxp0q/harvard_student_denied_entry_into_us_due_to/
-->

<!--
## Collaboration Policy

You are encouraged to discuss all labs, homeworks, and online quizzes with other students,
subject to the following constraints:

1. you must be the person typing in all code for your assignments, and
1. you may not look at another student's assignment.

You may use any online resources you like as references.
-->

<!--
## Self Grading

[An outlook on self-assessment of homework assignments in higher mathematics education](https://link.springer.com/article/10.1186/s40594-018-0146-z)

Also *Your* Job to Learn! Helping Students Reflect on their Learning Progress

Should you Allow your Students to Grade their own Homework?

Peer and Self Assessment in Massive Online Classes
-->

## Accommodations for Disabilities

I've tried to design the course to be as accessible as possible for all students.
If you need any further accommodations---even if you don't have an officially recognized disability---please ask.

I want you to succeed and I'll make every effort to ensure that you can.
