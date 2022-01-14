# -*- coding: utf-8 -*-
"""
Created on Tue May 18 23:22:56 2021

@author: Artem
"""

# General purpose
import pandas as pd
import numpy as np
import pickle
import os
import re
pd.options.mode.chained_assignment = None 
import pymorphy2
from stop_words import get_stop_words
from nltk.tokenize import word_tokenize
import pymorphy2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
stopwords = get_stop_words('ru')

def removeStopWords(string, stopwords=stopwords):

    result = [w for w in string.split() if w not in stopwords]
    result = " ".join(result)
    
    return result

# Read from pickle
os.chdir("/data")


### ---------------------------------------- 1. ISCO - Training dateset

# Loading and preprocessing
vacnames_isco = pd.read_excel("dictionary_isco.xlsx")
vacnames_isco.columns = ['name', 'isco']
vacnames_isco = vacnames_isco[vacnames_isco['isco'].notna()]
vacnames_isco['isco_3'] = vacnames_isco['isco'].apply(lambda x: str(x)[:3]) # 3-level of ISCO
vacnames_isco['isco_2'] = vacnames_isco['isco'].apply(lambda x: str(x)[:2]) # 2-level of ISCO
vacnames_isco['isco_1'] = vacnames_isco['isco'].apply(lambda x: str(x)[:1]) # 1-level of ISCO

# Add isco name for Second level
df_isco = pd.read_excel("isco-08-rus.xlsx")
df_isco['isco'] = df_isco['isco'].apply(
    lambda t: t.replace("Основная группа ", "").replace(":", ""))
df_isco['isco_3'] = df_isco['isco'].apply(lambda x: (re.findall('\d+', x))[0])
df_isco['isco_name'] = df_isco['isco'].apply(lambda x: ''.join(i for i in x if not i.isdigit()))
df_isco['isco_name'] = df_isco['isco_name'].apply(lambda x: x[1:])
df_isco = df_isco[df_isco['isco_3'].apply(lambda x: len(x) == 3)] # 3-level of ISCO
df_isco = df_isco[['isco_name', 'isco_3']]
vacnames_isco = vacnames_isco.merge(df_isco)

# Remove areas that are not present in HeadHunter
vacnames_isco = vacnames_isco[vacnames_isco['isco_3'].apply(
    lambda code: code not in {"111", "131", "231", "223"
                              "335", "521", "421", "323",
                              "262", "952", "631", "754"})]


vacnames_isco = vacnames_isco[vacnames_isco['name'].apply(lambda name: 'Федерац' not in name)]
vacnames_isco = vacnames_isco[vacnames_isco['name'].apply(lambda name: 'Государственн' not in name)]
vacnames_isco = vacnames_isco[vacnames_isco['name'].apply(lambda name: 'федеральн' not in name)]

# Text preprocessing
vacnames_isco['name'] = vacnames_isco['name'].apply(
    lambda name: name.replace('(руководитель)', 'руководитель'))
vacnames_isco['name'] = vacnames_isco['name'].apply(
    lambda name: name.replace('(врач)', 'врач'))
vacnames_isco['name'] = vacnames_isco['name'].apply(
    lambda name: name.replace('(заведующий)', 'заведующий'))
vacnames_isco['name'] = vacnames_isco['name'].apply(
    lambda name: re.sub(r'\([^)]*\)', '', name))
vacnames_isco['name'] = vacnames_isco['name'].apply(
    lambda name: re.sub(r'-', ' ', name))
vacnames_isco['name'] = vacnames_isco['name'].apply(
    lambda t: re.sub(r'[^A-Za-zА-Яа-я]+', ' ', t.lower()).strip())
vacnames_isco = vacnames_isco[vacnames_isco['name'] != '']
vacnames_isco['name'] = vacnames_isco['name'].apply(removeStopWords)

# Lemmatization
morph = pymorphy2.MorphAnalyzer()
vacnames_isco['name'] = vacnames_isco['name'].apply(
    lambda name: " ".join([morph.parse(t)[0].normal_form for t in name.split()]))

# Remove noisy words
vacnames_isco['name'] = vacnames_isco['name'].apply(lambda name: name.replace("ведущий", ""))
vacnames_isco['name'] = vacnames_isco['name'].apply(lambda name: name.replace("младший", ""))
vacnames_isco['name'] = vacnames_isco['name'].apply(lambda name: name.replace("старший", ""))

# TF-IDF vectorizer
tfidf_model = TfidfVectorizer(min_df=1, ngram_range=(1,2))
tfidf_matrix = tfidf_model.fit_transform(list(vacnames_isco['name']))
train_labels = vacnames_isco['isco_name']

# Classification with SVM
svmClassifier = OneVsRestClassifier(LinearSVC(), n_jobs=-1)
svmClassifier.fit(tfidf_matrix, train_labels)
svmPreds = svmClassifier.predict(tfidf_matrix)
f1_score(train_labels, svmPreds, average="micro")


### ---------------------------------------- 2. HeadHunter - Unlabeled dataset

# Loading
vacnames_hh = pd.read_pickle("vacnames_freq.obj")

# Remove noisy words
vacnames_hh['name_preprocessed'] = vacnames_hh['name_preprocessed'].apply(lambda name: name.replace("ведущий", ""))
vacnames_hh['name_preprocessed'] = vacnames_hh['name_preprocessed'].apply(lambda name: name.replace("младший", ""))
vacnames_hh['name_preprocessed'] = vacnames_hh['name_preprocessed'].apply(lambda name: name.replace("старший", ""))
vacnames_hh = vacnames_hh[vacnames_hh['name_preprocessed'] != '']

# Translate most popular English words to Russian
vacnames_hh['eng_name'] = vacnames_hh['name_preprocessed'].apply(lambda name: bool(re.search(r'[A-Za-z]+', name)))
vacnames_hh_eng = vacnames_hh[vacnames_hh['eng_name']]
vacnames_hh_eng_tokens = pd.Series(np.concatenate(vacnames_hh_eng['name'].apply(
    lambda x: x.split(" ")).values)).value_counts()
vacnames_hh_eng_tokens = vacnames_hh_eng_tokens[list(map(lambda name: bool(re.search(r'[A-Za-z]+', name)), vacnames_hh_eng_tokens.index))]
dict_eng_rus = {"manager":"менеджер",
                "developer":"программист",
                "engineer":"программист",
                "hr":"мененджер кадр",
                "specialist":"специалист",
                "sales":"продажа",
                "frontend":"программист",
                "backend":"программист",
                "analyst":"аналитик",
                "devops":"программист",
                "marketing":"маркетинг",
                "business":"бизнес",
                "head":"главный",
                "designer":"дизайнер",
                "assistant":"ассистент",
                "director":"директор",
                "consultant":"консультант",
                "intern":"стажер",
                "artist":"артист",
                "administrator":"администратор",
                "data":"программист",
                "call":"колл",
                "pr":"пи ар",
                "teacher":"учитель",
                "writer":"писатель",
                "techical":"технический",
                "accountant":"бухгалтер",
                "animator":"аниматор",
                "financial":"финансовый",
                "controller":"контролер",
                "recruiter":"рекрутер",
                "chief":"главный",
                "automation":"автоматизация",
                "architect":"программист",
                "driver":"водитель",
                "management":"менеджер",
                "ai":"программист",
                "receptionist":"администратор",
                "lawyer":"юрист",
                "system":"системный",
                "copywriter":"писатель"}
vacnames_hh['name_preprocessed'] = vacnames_hh['name_preprocessed'].apply(lambda name:
                                                ' '.join([dict_eng_rus.get(i, i) for i in name.split()]))

# Transform to TF-IDF based on the isco model
tfidf_matrix_hh = tfidf_model.transform(list(vacnames_hh['name_preprocessed']))

# ISCO class prediction
svmPreds_hh = svmClassifier.predict(tfidf_matrix_hh)
pd.Series(svmPreds_hh).value_counts()
vacnames_hh['isco_3'] = svmPreds_hh

# Quality of predictions
vacnames_hh['svm_proba'] = pd.DataFrame(
    svmClassifier.decision_function(tfidf_matrix_hh)).max(1).values

# OOV names
vacnames_hh['OOV'] = (pd.DataFrame(tfidf_matrix_hh.sum(1)) == 0)[0].values

### ---------------------------------------- 3. Add new names to the training dataset

# Add new names to the training dataset
vacnames_hh_excel = pd.read_excel("vacnames_hh_coded_iter_3.xlsx")
vacnames_hh_excel_match = vacnames_hh_excel[vacnames_hh_excel['mismatch'] != 1]
vacnames_hh_excel_match = vacnames_hh_excel_match[['name_preprocessed', 'isco_3_name', 'mismatch']]
vacnames_hh_excel_match = vacnames_hh_excel_match.drop(columns="mismatch")
vacnames_hh_excel_match.columns = ['name', 'isco_name']
t = vacnames_hh_excel[['isco_3_name', 'isco_3_code']].drop_duplicates()
t['isco_3_code'] = t['isco_3_code'].astype(str)
df_train = vacnames_isco[['name', 'isco_name']]
df_train = pd.concat([df_train, vacnames_hh_excel_match])

# TF-IDF vectorizer
tfidf_model = TfidfVectorizer(min_df=1, ngram_range=(1,2))
tfidf_matrix = tfidf_model.fit_transform(list(df_train['name']))
train_labels = df_train['isco_name']

# Classification with SVM
svmClassifier = OneVsRestClassifier(LinearSVC(), n_jobs=-1)
svmClassifier.fit(tfidf_matrix, train_labels)
svmPreds = svmClassifier.predict(tfidf_matrix)
f1_score(train_labels, svmPreds, average="micro")


### ---------------------------------------- 4. Get ISCO for Names for all

# Transform to TF-IDF based on the isco model
tfidf_matrix_hh = tfidf_model.transform(list(vacnames_hh['name_preprocessed']))

# ISCO class prediction
svmPreds_hh = svmClassifier.predict(tfidf_matrix_hh)
vacnames_hh['isco_3_name'] = svmPreds_hh
vacnames_hh = vacnames_hh.drop(columns="isco_3")
vacnames_hh = vacnames_hh.merge(df_isco,
                                left_on='isco_3_name', right_on='isco_name', how='left')
vacnames_hh = vacnames_hh.drop(columns="isco_name")
vacnames_hh = vacnames_hh.rename(columns={"isco_3":"isco_3_code"})

# Save to pickle
vacnames_hh.to_pickle("vacname_occupation.obj")

### ---------------------------------------- 5. Compare with ISCO stats

df_isco_stats = pd.read_excel("stats_isco.xlsx")
df_isco_stats = df_isco_stats[df_isco_stats['okz nas3'] != "Total"]
df_isco_stats = df_isco_stats[['okz nas3', 'Freq.']]
df_isco_stats.columns = ['isco_3_code', 'stat_freq']
df_isco_stats['isco_3_code'] = df_isco_stats['isco_3_code'].astype(str)

df_isco_comparision = pd.DataFrame(vacnames_hh.groupby(['isco_3_name', 'isco_3_code'])['freq'].sum())
df_isco_comparision = df_isco_comparision.reset_index()
df_isco_comparision['isco_3_code'] = df_isco_comparision['isco_3_code'].astype(str)

df_isco_comparision = df_isco_comparision.merge(df_isco_stats, how="right")
df_isco_comparision['rate'] = df_isco_comparision['stat_freq'] / df_isco_comparision['freq']
df_isco_comparision['stat_freq'] = df_isco_comparision['stat_freq'] / 1000
df_isco_comparision['freq'] = df_isco_comparision['freq'] / 1000
df_isco_comparision.corr()

df_isco_comparision['freq'].max() / df_isco_comparision['freq'].sum()






