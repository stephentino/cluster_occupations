
# Defining labor markets using Natural Language Processing (NLP) and task descriptions

## About this repository

The purpose of this project is to improve upon existing definitions of labor markets by using methods from Natural Language Processing (NLP). I define labor markets in a novel way by clustering occupations based on their task descriptions.

**Contents**

- [Background information](#Background-information)
- [Data](#Data)
- [Methodology](#Methodology)
- [Overview of the code](#Code)
- [Results](#Results)

## Background information

At the core of many papers in labor economics is the definition of the labor market. For example,
in a seminal work on the effects of immigration, [Borjas (2003)](https://doi.org/10.1162/003355303322552810) argued that labor markets are
appropriately defined as groups of workers with similar education and experience. Borjas’
conclusions about the effects of immigration in that paper depend heavily on the labor market
definition. Other studies in the immigration literature (e.g., [Ottaviano et al., 2013](https://doi.org/10.1257/aer.103.5.1925)) implicitly define
labor markets at the industry-year level. However, it is possible that true labor market
boundaries do not perfectly overlap with standard industry classification, and that researchers would arrive at different conclusions in their studies if they adopt different definitions of the
labor market in their analyses.

In this project, I use methods from Natural Language Processing (NLP) to cluster occupations based on their
task descriptions. My ultimate goal is to define a "labor market". Specifically in my context, a "labor market" is a
cluster of occupations for which (a) the tasks of all of the occupations are similar and (b) the
workers of these occupations have similar abilities, education, and experience.

## Data

This project uses data from the following two sources:

1. [O\*NET database](https://www.onetcenter.org/database.html) – The O\*NET database
contains rich information on over 900 occupations. It includes information on the skills,
abilities, and knowledge associated with each occupation, as well as each occupation’s
typical activities and tasks. It also includes information on the day-to-day aspects of many
jobs and the qualifications of the typical worker in each job.

2. [GloVe: Global Vectors for Word Representation (Pennington et al., 2014)](https://doi.org/10.3115/v1/D14-1162)) – the GloVe
database contains 1.9 million 300-dimensional vector representations of word meanings.
These can be used to map words to a vector space so that the similarity of two words can
assessed. The advantage of the GloVe database is that it can be used to identify words that
are similar even when they are not synonyms (e.g. “dog” and “puppy”). This database will
allow me to assess the similarity of task descriptions across occupations.

I also used stop words from the following sources (following Kogan et al. 2020):
- https://pypi.python.org/pypi/stop-words
- https://dev.mysql.com/doc/refman/5.1/en/fulltext-stopwords.html
- http://www.lextek.com/manuals/onix/stopwords1.html
- http://www.lextek.com/manuals/onix/stopwords2.html
- https://msdn.microsoft.com/zh-cn/library/bb164590
- http://www.ranks.nl/stopwords
- http://www.text-analytics101.com/2014/10/all-about-stop-words-for-text-mining.
- http://www.webconfs.com/stop-words.php
- http://www.nltk.org/book/ch02.html

The above sources produced 1,324 unique stop words.

## Methodology

I follow Kogan et al. (2020) to calculate the similarity of occupations by measuring the similarity
of their task descriptions. The O*NET database contains descriptions of the typical tasks of over
900 occupations. These descriptions are written in full English sentences. For example, there are
28 task descriptions for “Registered Nurse”, such as “record patients' medical information and
vital signs” and “monitor, record, and report symptoms or changes in patients' conditions”.
The first thing I did was lemmatize and tokenize each occupation’s task description using the
NLTK package in python. Then, I removed all stop words, where the list of stop words came from
several sources (see section IV). Next, I mapped the task descriptions to 300 dimensional vectors
in the GloVe database using the procedure below.
Denote by $A_i$ the set of ``word vectors” in the task description of occupation 𝑖, and denote by $X_i$ the
weighted average of these:

$X_i = \sum_{x_k \in A_i} w_{ik} x_k,$

where $x_k$ is a 300-by-1 dimensional vector representation of a word in the task description for
occupation 𝑖 (the task descriptions are from O\*NET and the vector representations of words are obtained from the GloVe database), and $w_{ik}$ is a scalar. Here, $w_{ik}$ is the term-frequency-inversedocument-
frequency (TFIDF) weight defined as

$w_{ik} = TF_{ik} \times IDF_k,$

where $TF_{ik} = \frac{c_{ik}}{\sum_j c_{ij}},$ with $c_{ij}$ denotes the count of the $j$th word in the task description of $i$, and $IDF_k$ is the natural log of the ratio of the number of occupations in the sample to the number of occupations in the sample with a task description that includes term $k$.

The above method produces a 300-dimensional real-valued vector for each occupation i in the O\*NET database.

After applying the above procedure, I obtained a 300-dimensional vector representation for each occupation in my data. I then used AgglomerativeClustering from sklearn.cluster to cluster the occupations into 50 clusters. The number 50 is arbitrary here, but it seems to work well. I used cosine similarity2 as the affinity in AgglomerativeClustering.

## Overview of the code

There is one main python script "*code/main.py*". This script accomplishes the following:
- preprocesses the task descriptions by removing stop words and punctuation
- creates TF-IDF weights using *sklearn*
- creates vector representations for each task description
- clusters the occupations to define labor markets

## Results

The results are available in "*results/occupation_cluster_results_April11.csv*". The "cluster" variable in that csv file defines the labor market associated with each occupation. If you would like to use this classification scheme to define labor markets, please cite this github and shoot me an email :) 

# Author
Stephen Tino, PhD Candidate in Economics, University of Toronto, s.tino@mail.utoronto.ca

