
# Defining labor markets using Natural Language Processing (NLP) and task descriptions

## About this repository

The purpose of this project is to improve upon existing definitions of labor markets by using methods from Natural Language Processing (NLP). I define labor markets in a novel way by clustering occupations based on their task descriptions.

**Contents**

- [Background information](#Background-information)
- [Data](#Data)

## Background information

At the core of many papers in labour economics is the definition of the labour market. For example,
in a seminal work on the effects of immigration, Borjas (2003) argued that labour markets are
appropriately defined as groups of workers with similar education and experience. Borjas’
conclusions about the effects of immigration in that paper depend heavily on the labour market
definition. Other studies in the immigration literature (e.g., Ottaviano et al., 2016) implicitly define
labour markets at the industry-year level. However, it is possible that true labour market
boundaries do not perfectly overlap with standard industry classification, and that researchers would arrive at different conclusions in their studies if they adopt different definitions of the
labour market in their analyses.

In this project, I use methods from Natural Language Processing (NLP) to cluster occupations based on their
task descriptions. My ultimate goal is to define a ``labor market”. Specifically in my context, a ``labor market” is a
cluster of occupations for which (a) the tasks of all of the occupations are similar and (b) the
workers of these occupations have similar abilities, education, and experience.

## Data

This project uses data from the following two sources:

1. O*NET database (https://www.onetcenter.org/database.html) – The O*NET database
contains rich information on over 900 occupations. It includes information on the skills,
abilities, and knowledge associated with each occupation, as well as each occupation’s
typical activities and tasks. It also includes information on the day-to-day aspects of many
jobs and the qualifications of the typical worker in each job.

2. GloVe: Global Vectors for Word Representation (Pennington et al., 2014) – the GloVe
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
- 
The above sources produced 1,324 unique stop words.

