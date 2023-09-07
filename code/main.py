# -*- coding: utf-8 -*-

# ---------------------------------------------------------------------------------------------------------------------------- #
# Copyright 2023 Stephen Tino                                                                                                  #
#                                                                                                                              #
# This file is part of cluster_occupations                                                                                     #
#                                                                                                                              #
# cluster_occupations is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General     #
# Public License as published by the Free Software Foundation, either version 2.1 of the License, or (at your option)          #
# any later version.                                                                                                           #
#                                                                                                                              #
# cluster_occupations is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied    #
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more          #
# details.                                                                                                                     #
#                                                                                                                              #
# You should have received a copy of the GNU Lesser General Public License along with estimate_akm. If not, see                #
# <https://www.gnu.org/licenses/>.                                                                                             #
# ---------------------------------------------------------------------------------------------------------------------------- #

import pandas as pd
import nltk
import numpy as np
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import matplotlib.pylab as plt
import scipy.sparse as sparse
from scipy import spatial
from stop_words import get_stop_words
import os

# change wd
os.chdir(r'C:\Users\Owner\Dropbox\3. UofT PhD\5. Winter 2021\ECO 3400\ECO 3400 Final Project - To hand in - Stephen Tino')


# I follow Kogan et al. 2020 Appendix A.1 for text preprocessing

#df = pd.read_excel(r'C:\Users\Owner\Dropbox\19. US Data\ONET\Task Statements.xlsx')
# Task Statements from ONET version 20.1 (uses 2010 code classification system)
df = pd.read_excel('Task Statements 2010 codes.xlsx')

sep='.'
df['O*NET-SOC Code'] =df['O*NET-SOC Code'].map(lambda x: x.split(sep,1)[0]) 

# make lower case
df['Task'] = df['Task'].str.lower()

### create list of stopwards to remove
# import punctuation
from string import punctuation
lstpunc = [char for char in punctuation]

# import stopwords
from nltk.corpus import stopwords
stop = set(stopwords.words('english'))
lststop=list(stop) #from set to list
pypi_stopwards = get_stop_words('english') # from pypi 

# stop words from other sources (see Kogan et al Appendix A1)
text_files = ['mysql_list.txt','minimal_stop.txt',
              'lextex_list1.txt','lextex_list2.txt',
              'microsoft_list.txt','rank_list1.txt',
              'rank_list2.txt','rank_list3.txt',
              'rank_list4.txt','snowball_list.txt',
              'terrier_stop.txt','webconfs_list.txt']

other_stopwards = []
for txtfile in text_files:
    my_file = open(txtfile, "r")
    for line in my_file:
        for word in line.split():
            other_stopwards.append(word)

# combine all of the above lists
lstremove = lststop + pypi_stopwards + other_stopwards + lstpunc
lstremove.append("'s")

# convert to set than back to list to remove duplicates
setremove = set(lstremove)
lstremove = list(setremove)

print('Number of (unique) stop words:',len(lstremove))


### Clean up task descriptions

# merge task descriptions into one entry for each SOC code
df['task_concat'] = df.groupby(['O*NET-SOC Code'])['Task'].transform(lambda x : ' '.join(x))
df_collapsed = df[['O*NET-SOC Code','task_concat']].drop_duplicates()

# remove stopwards/punctuation and clean up task descriptions
def remove_stopwards(df,lstremove):
    # create empty column as placeholder
    df['task_cleaned'] = None
    # prepare lemmatizer function
    lemma=WordNetLemmatizer()
    # loop through all occupations in the df
    for i in range(len(df)):
        temp_task = df['task_concat'].iloc[i]
        # tokenize
        token = nltk.word_tokenize(temp_task)
        # remove stopwards
        token_cleaned = [word for word in token if not word in lstremove]
        # lemmatize
        lem_token = []
        for j in range(len(token_cleaned)):
            lem_token.append(lemma.lemmatize(token_cleaned[j]))
        # concatenate everything together into one string
        temp_concat = ' '.join(token_cleaned)
        df['task_cleaned'].iloc[i] = temp_concat
    print('Successfully removed',len(lstremove), 'stopwords...')
    return df

# apply remove_stopwards function to my dataframe
df_cleaned = remove_stopwards(df_collapsed,lstremove)
documents = pd.Series.tolist(df_cleaned['task_cleaned'])

# check order preserved
for k in range(len(documents)):
    check = df_cleaned['task_cleaned'].iloc[k] == documents[k]
    if check == False:
        print('Error')

###

# construct DT matrix weighted by TF-IDF weights using sklearn
# For info on the default TF-IDF weighting in sklearn see section 6.2.3.4 in https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction
# I use mostly default settings except smooth_idf = False; this corresponds closely to Kogan et al.  (only difference is an extra +1 in the IDF computation)
vectorizer = TfidfVectorizer(use_idf=True,smooth_idf=False,max_df=0.5)

# fit model and create weighted DT matrix
dt_mat= vectorizer.fit_transform(documents)

# extract list of words as a list
vectorizer_tokens = vectorizer.get_feature_names()

# extract list of words as a dictionary
tokens_dict = vectorizer.vocabulary_

# extract idf weights
idf_temp = vectorizer.idf_

# convert to dataframe
dt_df = pd.DataFrame(data = dt_mat.toarray(),index = pd.Series.tolist(df_cleaned['O*NET-SOC Code']) ,columns = vectorizer_tokens)

###

# import GloVe
embeddings_dict = {}
print('Importing GloVe database...')
with open('glove.42B.300d.txt', 'r', encoding="utf-8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], "float32")
        embeddings_dict[word] = vector

print('Done importing GloVe...')
# function to find closest word embeddings
def find_closest_embeddings(embedding):
    return sorted(embeddings_dict.keys(), key=lambda word: spatial.distance.euclidean(embeddings_dict[word], embedding))

find_closest_embeddings(embeddings_dict["financial"])[:5]
        
###

# subset of GloVe database with words that appear in my task descriptions only
embeddings_dict_sub = {k: embeddings_dict[k] for k in embeddings_dict.keys() & vectorizer_tokens}
len(embeddings_dict_sub.keys())
len(embeddings_dict.keys())

# SOC codes
soc_codes = pd.Series.tolist(df_cleaned['O*NET-SOC Code'])

# function to create the weighted average of the vector representations for each document
# the output will be used to create the matrix of cosine similarity between SOC codes
def create_document_dictionary(word_list,document_list,e_dictionary,dt_df):
    doc_dict = {}
    for j in range(len(document_list)):
        temp_vec = np.zeros(300)
        # Loop through all words 
        for i in range(len(word_list)):
            word_i = word_list[i]
            if word_i in list(e_dictionary.keys()):
                # cumulative sum of weight * vectorization of word based on word embeddings in GloVe, where the weights from the TF-IDF-weighted document-word count matrix
                temp_vec = temp_vec + dt_df.iloc[j][word_i]* e_dictionary[word_i]
            else:
                continue
        # add entry to dictionary 
        doc_dict[document_list[j]]=temp_vec
        if (j % 10 == 0):
            print('creating dictionary entry',j,'out of',len(document_list),'...')
        else:
            continue
    return doc_dict
    
# apply the above function to create a dictionary where the key is the SOC code and the entry is the 300-dimensional vector representation of the SOC code
soc_dict = create_document_dictionary(vectorizer_tokens,soc_codes,embeddings_dict_sub,dt_df)

# create a dictionary that maps SOC codes to occupation titles
df_soc_title = df[['O*NET-SOC Code','Title']].drop_duplicates()
df_soc_title['Title'] = df_soc_title.groupby(['O*NET-SOC Code'])['Title'].transform(lambda x : '/'.join(x))
df_soc_title = df_soc_title.drop_duplicates()

soc2occ = {}
for i in range(len(df_soc_title)):
    soc2occ[df_soc_title['O*NET-SOC Code'].iloc[i]] = df_soc_title['Title'].iloc[i]

# create a dictionary that maps occupation titles to SOC codes
occ2soc = {}
for i in range(len(df_soc_title)):
    occ2soc[df_soc_title['Title'].iloc[i]] = df_soc_title['O*NET-SOC Code'].iloc[i]
    
#df_soc_title[df_soc_title['Title'] == 'Stockers and Order Fillers']
#df_soc_title[df_soc_title['Title'] == 'Tank Car, Truck, and Ship Loaders']

# function to find closest occupations:
def find_closest_occupations(occ,input_format,number):
    if input_format == 1:
        soc = occ # input_format == 1 => already soc format
        return sorted(soc_dict.keys(), key=lambda word: spatial.distance.euclidean(soc_dict[word], soc_dict[soc]))[:number]
    if input_format == 2:
        soc = occ2soc[occ] # convert to SOC code since input_format == 2 => title format (calling this 'occ')
        soc_output = sorted(soc_dict.keys(), key=lambda word: spatial.distance.euclidean(soc_dict[word], soc_dict[soc]))[:number]
        occ_output = []
        for i in range(len(soc_output)):
            occ_output.append(soc2occ[soc_output[i]]) # convert all output to occ format
    return occ_output

print(find_closest_occupations('Bakers',2,10))
       
print(find_closest_occupations('Flight Attendants',2,10))

# store dictionary as a pandas DF
df_embeddings = pd.DataFrame(soc_dict)
#soc_dict['11-1011.03']
# transpose
df_embeddings_T = df_embeddings.T

# extract array
data = df_embeddings_T.values 

from sklearn.cluster import AgglomerativeClustering

cluster = AgglomerativeClustering(n_clusters=50, affinity='cosine', linkage='average')
df_embeddings_T['cluster'] = cluster.fit_predict(data)
df_embeddings_T['SOC'] = df_embeddings_T.index
df_embeddings_T['Title'] = df_embeddings_T['SOC'].map(lambda x: soc2occ[x])
results = df_embeddings_T[['SOC','Title','cluster']]


results = results.sort_values(by=['cluster'])
# need to add occ descriptions and done

#cos_df.to_csv('SOC_cosine_similary.csv',index=False)

results.to_csv('occupation_cluster_results_April11.csv',index=False)

# compute pairwise cosine similarity
from sklearn.metrics.pairwise import cosine_similarity
cos_df = cos_df = pd.DataFrame(cosine_similarity(df_embeddings.T), columns = df_embeddings.columns)
cos_df.insert(0,"SOC code",df_embeddings.columns)

# export file

#cos_df.to_csv('SOC_cosine_similary.csv',index=False)
cos_df.to_csv('SOC_cosine_similary_2010codes.csv',index=False)

# end

