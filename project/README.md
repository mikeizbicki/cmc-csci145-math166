# Project

**Overview:** 
You will apply data mining to a dataset of your choice,
write up your results,
and share them with the world on social media.
This project will teach you about data mining and technical writing,
and you can add it to your "portfolio" to show future potential employers.

## Submissions

The project is divided into three stages: proposal, checkup, and final submission.

### Proposal

The proposal is due **Tuesday, 1 Oct** at the beginning of class.

The proposal has a written and an oral component.
The written component should be no longer than 1 page and contain the following sections:

1. Working title

1. List of project members:  You may work in teams of 1-2 people.  (Larger teams are possible with prior approval from me for sufficiently challenging projects.)

1. Research question/hypothesis: A one sentence statement about what you will be investigating.

1. Background: A 1 paragraph description of why your research question/hypothesis is important.

1. Dataset: A 1-2 paragraph description of the dataset you will use.  Be sure to include information about how you will collect the data, and why the dataset will be appropriate for answering your research question.

1. Publication venues: A list of at least 5 social media communities that you believe will be interested in the results of your project, along with a 1 sentence description about why.  Examples include subreddits, websites like hacker news, and communities on twitter.

For the oral component,
each team will explain to the rest of the class the goals of their project.
This explanation should be about 1 minute.

### Checkup

The checkup is due **Thursday, 14 Nov** at the beginning of class.

The purpose of the checkup is to ensure that you are making adequate progress towards your final project.
You should be at least 60% complete with all the technical work of the project.
At a minimum, you should:

1. have access to the precise data you will be using (for example if it's twitter data, you should have filtered it down to the exact set of tweets),

1. have at least 3 figures generated, and

1. have several sentences for each figure that explain the key insights of the figure.

It should be obvious to both you and me exactly what work you have left to complete.

Your checkup submission will include:

1. an html document that is an incomplete draft of your final submission, and

1. a short presentation (5ish minutes) explaining the key insights from each of your 3 graphs and what work you have left to do.

### Final Submission

The final submission is due **Tuesday, 10 Dec** at the beginning of class.

Your final submission will be graded according to the following rubric.

| Category              | Points  | 
| --------------------- | ------- |
| Figures               | 50      |
| Title                 | 5       |
| tl;dr                 | 10      |
| Body of the article   | 20      | 
| Technical appendix    | 15      |

**Figures:**

A standard final project will have 5 figures, with each figure worth 10 points each.
Additional figures beyond the 5 are eligible for up to 10 points of extra credit each.
Some projects may have fewer than 5 figures if the figures are particularly informative or difficult to generate.
If you believe your project should have fewer than 5 figures, 
then you must get this approved by me by the project checkup.

Each figure will be graded on two criteria:
Half the points are for the technical quality of the figure.
Technical issues you should address include:

1. Are the axes/lines appropriately labeled?
1. Are the ranges of the axes appropriate?
1. Is the choice of bar/line/etc plots appropriate?
1. Is the text in the figure appropriately labelled?
1. Is the figure of an appropriate resolution?

The other half of the points are for the new insights your figure generates.
Every figure must teach me something I did not previously know.

You must have at least 1 figure that:

* shows "what the data looks like"

* provides insight about the data using simple statistics

* shows the results of your estimation procedure

**Writing:**

As with your figure grade, half of your writing grade is based on the technical quality of your writing, and half is based on the new insights that your writing conveys.

Your writeup should satisfy the following requirements:

1. The title should highlight the key insight of your post in 1 phrase
1. The tl;dr should succinctly explain the key insights of your post in 1-4 sentences
1. The body should:
    1. clearly introduce and explain each figure
    1. include appropriate references and footnotes (approximately 2 footnotes and 4 hyperlinks would be typical, although what is appropriate will vary per project)
    1. have a clear "story"
    1. should be divided into appropriate sections
    1. **be readable by a motivated English major**
1. The technical appendix should include sufficient details of your estimation procedure for me to reproduce it exactly; it should also include sufficient explanations of techniques that a math/cs major who hasn't taken data mining would understand what you did

Your write-up must clearly identify at least one of the sources of error discussed in class (estimation error, optimization error, etc.), how that error relates to your problem in an interesting way, and what you did to address that error.
You should include as much of this explanation as possible in the main body of your write-up, 
but parts of this explanation may need to appear in a footnote or in the appendix.

## Examples

I have a few "theoretical" examples on my personal blog:

1. [How to cheat at settlers of Cataan by loading the dice](https://izbicki.me/blog/how-to-cheat-at-settlers-of-catan-by-loading-the-dice-and-prove-it-with-p-values.html)

1. [How to create an unfair coin and prove it with math](https://izbicki.me/blog/how-to-create-an-unfair-coin-and-prove-it-with-math.html)

Some other examples:

1. DigitalNK: [Gender and North Korean posters](https://digitalnk.com/blog/2017/09/30/gender-distribution-in-north-korean-posters/)

1. DigitalNK: [North and South Korea Through Word Embeddings](https://digitalnk.com/blog/2017/12/23/north-and-south-korea-through-word-embeddings/)

1. DigitalNK: [Visualizing the Korean War: Data Bombs and Propaganda](https://digitalnk.com/blog/2017/10/08/visualizing-the-korean-war-bombs-propaganda-and-data-visualization/)

1. OkCupid: [don't be ugly by accident](https://theblog.okcupid.com/dont-be-ugly-by-accident-b378f261dea4)

1. OkCupid: [exactly what to say in a first message](https://theblog.okcupid.com/exactly-what-to-say-in-a-first-message-2bf680806c72)

1. OkCupid: [the big lies people tell in online dating](https://theblog.okcupid.com/the-big-lies-people-tell-in-online-dating-a9e3990d6ae2)

1. [NLP analysis of net neutrality comments](https://medium.com/hackernoon/more-than-a-million-pro-repeal-net-neutrality-comments-were-likely-faked-e9f0e3ed36a6)

**NOTE:**
These are all examples of how to present data analysis online.
Many of these examples are much more in depth than you will need to do.

## Topic ideas

1. The Claremont Colleges have several school newspapers (e.g. The Student Life, The CMC Forum, The Claremont Independent, The Scripps Voice).  Can we use data mining techniques to quantify the editorial bias of each of these newspapers?

1. I have access to billions of tweets sent from around the world.  Can we determine which topics CMC students are tweeting about the most?  How does this compare to Mudd/Scripps/Pitzer/Pomona students?

1. I have access to all reddit posts ever created.  Given a small set of related subreddits, can we determine what the distinguishing features of these subreddits are?

1. Prof Hanzhang Liu (Pitzer, Political Studies) has a dataset of Chinese newspaper articles about Chinese relations.  Can we use these articles to infer which countries China has close relations with?

1. Prof Gabriel Cook (CMC, Psychology) has a dataset of student written stories that are labeled by the emotions described in the stories.  Can we create a system that automatically predicts emotions from stories?
