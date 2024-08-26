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

<img src=img/math.webp width=600px />

<!--
<center>
<img width='400px' src='img/machine_learning_2x.png' />
</center>
-->

## About the Instructor

|||
|-|-|
| Name | Mike Izbicki (call me Mike) |
| Email | mizbicki@cmc.edu |
| Office | Adams 216 |
| Office Hours | TBA |
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
1. taught in [DPRK](https://pust.co)

## About the Course

**General Information:**

1. This is the theory course for CMC's Data Science major
1. Prepare you for industry or graduate school
    1. Especially for machine learning technical interviews
    1. No SQL in this course => that's CSCI143 Big Data

**Course Content:**

See the [introduction Jupyter notebook](intro.ipynb)

This course is divided into three sections:

1. *Pagerank*

    We will use the "Deeper Inside Pagerank" paper.

    <img src=img/deeper-inside-pagerank.jpg />

    You will learn:
    1. How to measure the runtime of data mining algorithms
    1. Trade-offs in runtime and "accuracy"
    1. How eigenvectors are used in data mining

1. *Machine Learning*

    We will cover everything in *Learning from Data* by Yaser S. Abu-Mostafa, Malik Magdon-Ismail, and Hsuan-Tien Lin.

    <img src=img/book.jpg width=200px />

    > **NOTE:**
    > I will provide you all a free copy.

    You will learn:
    1. Techniques
        1. Logistic regression
        1. Kernel methods
        1. Neural networks
    1. Math
        1. Bias/variance trade-off
        1. VC Dimension theorem (fundamental theorem of statistical learning)
        1. Regularization (L1, L2, elastic net, weight decay, early stopping, etc.)
        1. Optimization algorithms (gradient descent, stochastic gradient descent, ADAM, etc.)

1. *Applications*

    We will use a variety of papers/blog posts.
    Focus on text/web/social media examples.

**Other Notes:**

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

**Grades:**

| Category                          | Points        |                               |
| --------------------------------- | ------------- | ----------------------------- |
| Projects                          | `2**2`-`2**3` | every 2-3 weeks, 5 in total   |
| Quizzes                           | `2**2`-`2**3` | as needed                     |
| Midterm 1 (Pagerank)              | `2**5`        | (approximately) 23 Sep        |
| Midterm 2 (Learning from Data)    | `2**5`        | (approximately) 01 Nov        |
| Final                             | `2**7`        |                               |

*Late Work Policy:*

1. You lose `2**(i-1)` points on every assignment,
    where `i` is the number of days late.

1. Do not expect partial credit for incomplete assignments.
    It is much better to submit a correct assignment late than an incorrect one on time.

1. I expect most people to get full credit on the assignments.

*Exams:*

1. Majority of your grade
1. No programming, only math
1. Oral exams, 1-1 in my office
1. Final exam can replace all previous exams


**This is a hard class:**

1. The material is intrinsically hard.

    1. Very few people find linear algebra, statistics and programming to ALL be easy subjects, and this class combines them all.
    1. There's a reason people who understand this material get paid big salaries at FAANG.

1. You will have to read the required references.

    Not all the material will be covered in lectures,
    and that's intentional to force you to get practice reading research-level data mining text.

    > **NOTE:**
    > In all of my other courses, I include required reading/watching tasks to learn about CS/DS culture.
    > This course doesn't have these tasks because there is already a LOT of textbook reading that you will have to complete.

1. Comments from previous students:

    1.  > Holy fucking shit this was a hard class.
        > I had no idea there was so much god damned fucking math involved in a CS class.
        > You should warn students about that.

        Unfortunately, I can't remove the math from this class,
        and I can't make the class easier.
        Otherwise, you wouldn't be learning the material needed to pass a technical interview / get a good job / go to grad school.

    1.  > I got my job because of the data mining course.
        > Technical interviews were super easy after this class.

        What do data scientists get paid?
        1. <https://www.levels.fyi/t/data-scientist?countryId=254>
        1. <https://www.bls.gov/ooh/math/data-scientists.htm>

**Collaboration Policy:**

You are encouraged to discuss all projects with other students,
subject to the following constraints:

1. you must be the person typing in all code for your assignments, and
1. you must not copy another student's code.

You may use any online resources you like as references.

Basically, I'm trusting you all to be adults.
You are ultimately responsible for ensuring you learn the material!
So do what will help you learn best.

## Accommodations for Disabilities

I've tried to design the course to be as accessible as possible for all students.
If you need any further accommodations---even if you don't have an officially recognized disability---please ask.

I want you to succeed and I'll make every effort to ensure that you can.
