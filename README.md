# CSCI145 / MATH166: Data Mining

## About the Instructor

|||
|-|-|
| Name | Mike Izbicki (you can call me either Mike or Prof Izbicki) |
| Email | mizbicki@cmc.edu |
| Webpage | [izbicki.me](https://izbicki.me) |
| Research | Machine Learning (see [izbicki.me/research.html](https://izbicki.me/research.html) for some past projects) |
| Office | Adams 216 |
| Office Hours | Tuesday/Thursday 2:40-4:10 or by appointment ([see my schedule](https://outlook.office365.com/owa/calendar/45eb28fd4e4f45f4b0d120d17676d937@ClaremontMcKenna.edu/a46ebec5e46b4328abcb964af38795935165582125062542146/calendar.html)) |
| Fun Facts | grew up in San Clemente, 7 years in the navy, phd/postdoc at UC Riverside, taught in DPRK |

I like having lunch with students and getting to know you better.
If you want to get lunch, use [my bookings page](https://outlook.office365.com/owa/calendar/MeetwithMike@claremontmckenna.onmicrosoft.com/bookings/) to schedule a time.

## About the Course

**Learning Objectives:**

1. Understand basic machine learning theory
    1. algorithms for regression/classification/clustering
    1. the bias/variance trade-off
1. Understand feature generation methods, especially for analyzing text and social media
1. Understand the ethical implications of data mining
1. Apply data mining libraries (scikit-learn, gensim, others based on class interest) to **real world applications**

This course will **not** have you implementing complex algorithms.
I will be offering a course next semester on deep learning and statistical learning theory that will cover algorithmic implications in more detail.

**Textbook:**

1. Christopher Bishop's *Pattern Recognition and Machine Learning*.  [Download a free pdf copy from Microsoft Research.](https://www.microsoft.com/en-us/research/people/cmbishop/prml-book/)
1. Peterson's *The Matrix Cookbook* is a handy reference for multivariable calculus, and [you can download a free copy here](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf).
1. Other articles as posted in the schedule below.
1. Other popular (and freely available) data mining books include [Data Mining: Concepts and Techniques](http://myweb.sabanciuniv.edu/rdehkharghani/files/2016/02/The-Morgan-Kaufmann-Series-in-Data-Management-Systems-Jiawei-Han-Micheline-Kamber-Jian-Pei-Data-Mining.-Concepts-and-Techniques-3rd-Edition-Morgan-Kaufmann-2011.pdf) by Jiawei Han, Micheline Kamber and Jian Pei and [Data Mining](http://www.charuaggarwal.net/Data-Mining.htm) by Charu C. Aggarwal.  We probably won't use these books at all though.

**Grades:**

| Category              | Percent |
| --------------------- | ------- |
| Homework              | 40      |
| Quizes                | 10      |
| Project (proposal)    | 10      |
| Project (checkup)     | 10      |
| Project (final)       | 30      |

If your project proposal or checkup grades are lower than your final project grade,
then I will replace those grades with your final project grade.

**Late Work Policy:**

You lose 10% on the assignment for each day late.
If you have extenuating circumstances, contact me in advance of the due date and I may extend the due date for you.

## Schedule

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

## Collaboration Policy

You are encouraged to discuss all labs, homeworks, and online quizzes with other students,
subject to the following constraints:

1. you must be the person typing in all code for your assignments, and
1. you may not look at another student's assignment.

You may use any online resources you like as references.

<!--
## Self Grading

[An outlook on self-assessment of homework assignments in higher mathematics education](https://link.springer.com/article/10.1186/s40594-018-0146-z)

Also *Your* Job to Learn! Helping Students Reflect on their Learning Progress

Should you Allow your Students to Grade their own Homework?

Peer and Self Assessment in Massive Online Classes
-->

## Accommodations for Disabilities

I want you to succeed and I'll make every effort to ensure that you can.
If you need any accommodations, please ask.

If you have already established accommodations with Disability Services at CMC, please communicate your approved accommodations to me at your earliest convenience so we can discuss your needs in this course. You can start this conversation by forwarding me your accommodation letter. If you have not yet established accommodations through Disability Services, but have a temporary health condition or permanent disability (conditions include but are not limited to: mental health, attention-related, learning, vision, hearing, physical or health), you are encouraged to contact Assistant Dean for Disability Services & Academic Success, Kari Rood, at disabilityservices@cmc.edu to ask questions and/or begin the process. General information and the Request for Accommodations form can be found at the CMC DOS Disability Service’s website. Please note that arrangements must be made with advance notice in order to access the reasonable accommodations. You are able to request accommodations from CMC Disability Services at any point in the semester. Be mindful that this process may take some time to complete and accommodations are not retroactive. It is important to Claremont McKenna College to create inclusive and accessible learning environments consistent with federal and state law. If you are not a CMC student, please connect with the Disability Services Coordinator on your campus regarding a similar process.

