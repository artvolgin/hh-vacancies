# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 23:39:37 2020

@author: Artem
"""

# =============================================================================
# 1. Libraries
# =============================================================================
# General purpose
import pandas as pd
import numpy as np
import math
import random
import sklearn
import pickle
import seaborn as sns
import scipy.stats as ss
import os
from itertools import combinations
import re
import time
from datetime import datetime
import matplotlib.pyplot as plt
import scipy
import sklearn.preprocessing as pp
import sklearn.metrics
import itertools
import gc
import math
import json
import urllib
import requests
from bs4 import BeautifulSoup
from statistics import mode
import sqlite3
from operator import or_
from functools import reduce # python3 required
import dateutil.parser
import io
pd.options.mode.chained_assignment = None 

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

# =============================================================================
# 1. Preprocessing functions
# =============================================================================

# Change the working directory
os.chdir('/data')

#####  Preprocessing


# Extract salary intervals
def getSalaryIntervals(salary_dict, interval_type):
    if salary_dict == None:
        salary_to = salary_from = np.nan
    # TODO: Convertations between currencies
    # Convert with https://api.hh.ru/dictionaries
    elif salary_dict['currency'] != 'RUR': 
        salary_to = salary_from = np.nan
    else:
        salary_from = salary_dict['from']
        salary_to = salary_dict['to']
        if salary_dict['gross'] and type(salary_from) == 'float':
            salary_from = salary_from*0.87
        if salary_dict['gross'] and type(salary_to) == 'float':
            salary_to = salary_to*0.87
    
    if interval_type == "from":
        result = salary_from
    else:
        result = salary_to
    
    return result


def batchPreprocessing(df_batch):

    # Remove missings
    df_batch = df_batch[df_batch['id'].notna()]
    
    # Select only unique vacancies 
    df_batch = df_batch.drop_duplicates('id')
    
    # Reset index
    df_batch.reset_index(drop=True, inplace=True)
    
    # Area
    df_batch['area_id'] = df_batch['area'].apply(lambda x: x["id"])
    df_batch['area_name'] = df_batch['area'].apply(lambda x: x["name"])
    
    # Date of vacancy creation
    df_batch['created_at_date'] = df_batch.created_at.apply(
        lambda d: dateutil.parser.parse(d))
    
    # Employment
    df_batch['employment'] = df_batch['employment'].apply(lambda d: d['id'])
    
    # Experience
    df_batch['experience'] = df_batch['experience'].apply(lambda d: d['id'])
    
    # Employer information
    df_batch['employer_id'] = df_batch['employer'].apply(
        lambda d: d['id'] if "id" in d else np.nan)
    df_batch['employer_name'] = df_batch['employer'].apply(
        lambda d: d['name'] if "name" in d else np.nan)
    
    # Salary intervals
    df_batch['salary_from'] = df_batch['salary'].apply(lambda x: getSalaryIntervals(x, 'from'))
    df_batch['salary_to'] = df_batch['salary'].apply(lambda x: getSalaryIntervals(x, 'to'))
    
    # Mean salary
    df_batch['salary_mean'] = df_batch.loc[:,['salary_from', 'salary_to']].apply(
        lambda interval: np.mean(interval), 1)
    
    # Schedule
    df_batch['schedule'] = df_batch['schedule'].apply(lambda d: d['id'])
    
    # Extract skills (also recode list to string for SQL, use "eval" to obtain the list)
    df_batch['key_skills'] = df_batch['key_skills'].apply(
        lambda ks: repr([s['name'] for s in ks] if len(ks) > 0 else np.nan))
    
    # Specialization and Profarea (also recode list to string for SQL, use "eval" to obtain the list)
    df_batch['specialization_id'] = df_batch['specializations'].apply(
        lambda vec: repr([d['id'] for d in vec]))
    df_batch['specialization_name'] = df_batch['specializations'].apply(
        lambda vec: repr([d['name'] for d in vec]))
    df_batch['profarea_id'] = df_batch['specializations'].apply(
        lambda vec: repr([d['profarea_id'] for d in vec]))
    df_batch['profarea_name'] = df_batch['specializations'].apply(
        lambda vec: repr([d['profarea_name'] for d in vec]))
    
    # Recode address dict to string, use "eval" to obtain the list
    df_batch['address'] = df_batch['address'].apply(lambda d: repr(d))
    
    # Drop redundant columns
    drop_columns = ["accept_incomplete_resumes", "allow_messages", "apply_alternate_url", "archived", "area",
                    'accept_temporary', 'working_days', 'working_time_intervals', 'working_time_modes',
    "billing_type", "branded_description", "code", "contacts", "department",
    "driver_license_types", "errors", "has_test", "hidden", "immediate_redirect_url",
    "insider_interview", "negotiations_url", "premium", "quick_responses_allowed",
    "relations", "request_id", "response_letter_required", "response_url", "site",
    "description", # <<< REMOVE DESCRIPTION FOR NOW
    "employer_name", 'accept_handicapped', 'accept_kids', # <<< NEW COLUMNS TO REMOVE
    'address', 'area_name', 'specialization_name', 'created_at', # <<< NEW COLUMNS TO REMOVE
    "suitable_resumes_url", "test", "type", "vacancy_constructor_template", "published_at",
    "specializations", "employer", "salary", "alternate_url"]
    drop_columns = set(df_batch.columns).intersection(set(drop_columns))
    df_batch = df_batch.drop(drop_columns, "columns")
    
    # Sort columns by names
    df_batch = df_batch[np.sort(df_batch.columns)]
    
    return df_batch


# =============================================================================
# 2. Load files to SQL
# =============================================================================

# Connect to SQLite
conn = sqlite3.connect("D:/hh_sql_db3/hh.db")
cur = conn.cursor()

# Read all file names from working directory 
files_in_dir = os.listdir("D:/hh_archive")
files_in_dir = list(filter(lambda file_name: file_name.startswith("df_"), files_in_dir))

for i in range(len(files_in_dir)):
    
    try: 
        # Load batch of 100,000 vacancies
        df_batch = pd.read_pickle(files_in_dir[i])
        
        if type(df_batch) != pd.DataFrame:
            # Remove missings
            df_batch = [e for e in df_batch if type(e) == dict]
            # Convert to dataframe
            df_batch = pd.DataFrame(df_batch)
        
        # Preprocessing
        df_batch = batchPreprocessing(df_batch)
        
        # Save to SQL
        df_batch.to_sql(files_in_dir[i], conn, if_exists="replace")
        
        # Print stats
        print(i, files_in_dir[i], df_batch.shape)
        
    except:
        print(i, files_in_dir[i], " -------- ERROR")


### Merge batch tables into one dataframe

# Rename first table
cur.execute("ALTER TABLE " + files_in_dir[0] + " RENAME TO df_hh;")

# Insert other batches into the first one
union_all_part = "(" + " UNION ALL ".join(["SELECT * FROM " + file_name for file_name in files_in_dir[1:]]) + ")"
cur.execute("INSERT INTO df_hh SELECT * FROM " + union_all_part)
conn.commit()
print("Merged in one table")

# Remove seperate batch tables
for file_name in files_in_dir[1:]:
    cur.execute("DROP TABLE " + file_name)
    conn.commit()
    print(file_name)

# =============================================================================
# 3. Add ADDRESS
# =============================================================================

# Set working directory
os.chdir('D:/hh_archive')

def batchPreprocessing_address(df_batch):

    # Select only id and description
    df_batch = df_batch.loc[:,['id','address']]
    # Remove missings
    df_batch = df_batch[df_batch['id'].notna()]
    # Select only unique vacancies 
    df_batch = df_batch.drop_duplicates('id')
    # Reset index
    df_batch.reset_index(drop=True, inplace=True)
    
    return df_batch


# Connect to SQLite
conn = sqlite3.connect("D:/hh_sql_db3/hh_address.db")
cur = conn.cursor()

# Read all file names from working directory 
files_in_dir = os.listdir("D:/hh_archive")
files_in_dir = list(filter(lambda file_name: file_name.startswith("df_"), files_in_dir))

df_hh_address = list()
for i in range(len(files_in_dir)):
    
    try: 
        # Load batch of 100,000 vacancies
        df_batch = pd.read_pickle(files_in_dir[i])
        
        if type(df_batch) != pd.DataFrame:
            # Remove missings
            df_batch = [e for e in df_batch if type(e) == dict]
            # Convert to dataframe
            df_batch = pd.DataFrame(df_batch)
        
        # Preprocessing
        df_batch = batchPreprocessing_address(df_batch)
        
        if len(df_batch) > 0:
            df_hh_address.append(df_batch)
        
        # Print stats
        print(i, files_in_dir[i], df_batch.shape)
        
    except:
        print(i, files_in_dir[i], " -------- ERROR")

# Close the connection
conn.close()

df_hh_address = pd.concat(df_hh_address, ignore_index=True)

# Save to pickle
os.chdir("/data")
# Dataset with ids and address
df_hh_address.to_pickle("df_hh_address.obj")


# =============================================================================
# 3. Preprocessing 2019-2020
# =============================================================================

# Intialize the connection to SQL
conn = sqlite3.connect("D:/hh_sql_db3/hh.db")
conn.text_factory = str
cur = conn.cursor()

# Load the dataset
start_time = time.time()
df_hh = pd.read_sql_query(("SELECT created_at_date, name, salary_mean, key_skills, " 
                          "area_id, profarea_name, specialization_id, "
                          "employer_id, experience, "
                          "schedule, employment, id " 
                          "FROM df_hh"), conn)
# Close the connection
conn.close()
print("--- %s minutes ---" % ((time.time() - start_time) / 60))


### 3.1 Time preprocessing
# Select the data from 2019 onwards
df_hh = df_hh[df_hh['created_at_date'] >= "2019-01-01"]
# Remove vacancies after January 1, 2021
df_hh = df_hh[df_hh['created_at_date'] < "2021-01-01"]
# Replace datetime with just date in YYYY-MM-DD format
df_hh['created_at_date'] = df_hh['created_at_date'].apply(
         lambda d: d[:10])
# Create column with date in YYYY-MM format
df_hh['created_at_month'] = df_hh['created_at_date'].apply(lambda d: d[:-3])
# Convert to datetime
df_hh['created_at_month'] = pd.to_datetime(df_hh['created_at_month'])
df_hh['created_at_date'] = pd.to_datetime(df_hh['created_at_date'])
# Obtain the date in YYYY-WEEK format
df_hh['created_at_week'] = df_hh['created_at_date'].apply(
    lambda d: str(d.isocalendar()[0]) + "_" + str(d.isocalendar()[1]))

### 3.2 Region preprocessing
cross_regions = df_hh['area_id'].value_counts()
cross_regions = pd.DataFrame(cross_regions).reset_index()
cross_regions.columns = ['area_id', 'vac_number']
# Load the list of areas id and corresponding names
areas_id_name = json.loads(requests.get("https://api.hh.ru/areas").text)
# Select only Russian areas
areas_id_name = areas_id_name[0]['areas'] 
# Append the coresponding regions to the areas
dict_areas_region = dict()
for region in areas_id_name:
    for area in region['areas']:
        dict_areas_region[area['id']] = region['name']
dict_areas_region['1'] = "Москва"
dict_areas_region['2'] = "Санкт-Петербург"
cross_regions['region_name'] = cross_regions['area_id'].replace(dict_areas_region)
# Save a set of non-russian areas id
nonrussian_areas = set(cross_regions[cross_regions['region_name'].apply(
    lambda n: n.isnumeric())]['area_id'].values)
# Remove non-russian regions from database
df_hh = df_hh[df_hh['area_id'].apply(lambda a_id: a_id not in nonrussian_areas)]
# Reset index
df_hh.reset_index(drop=True, inplace=True)

### 3.3 Salary preprocessing
# Remove outliers in the salary
df_hh['salary_mean'][df_hh['salary_mean'] <= np.nanquantile(df_hh['salary_mean'], 0.005)] = np.nan
df_hh['salary_mean'][df_hh['salary_mean'] >= np.nanquantile(df_hh['salary_mean'], 0.995)] = np.nan
# Create variable that indicates presence of the salary in the vacancy
df_hh['salary_present'] = df_hh['salary_mean'].notna().astype(int)

### 3.4 key_skills presence
df_hh["skills_present"] = df_hh['key_skills'].apply(lambda x: x != 'nan')
df_hh["skills_present"] = df_hh["skills_present"].astype(int)

### 3.5 Key skills preprocessing
def getEvalVariable(x):
    if x == 'nan':
        x = np.nan
    else:
        x = np.array(eval(x))
    return x

df_hh['key_skills'] = df_hh['key_skills'].apply(getEvalVariable)

### 3.6 Specializations preprocessing
df_hh['specialization_id'] = df_hh['specialization_id'].apply(getEvalVariable)
# Calculate specialization counts
vec_spec = df_hh['specialization_id']
vec_spec = vec_spec[vec_spec.notna()]
vec_spec = np.concatenate(vec_spec.values)
cross_spec = pd.Series(vec_spec).value_counts()
cross_spec = pd.DataFrame(cross_spec).reset_index()
cross_spec.columns = ['specialization_id', "vac_number"]

# Get the proper labels for the specialization ids from api.hh.ru
# Load the list of specialization ids and corresponding names
specialization_id_name = json.loads(requests.get(
    "https://api.hh.ru/specializations").text)
# Fix the error
del specialization_id_name[0]['specializations'][-1]
# Create the dataframe with id and name of the profarea
df_profarea = [{"profarea_id":profarea["id"], "profarea_name":profarea["name"]} for profarea in specialization_id_name]
df_profarea = pd.DataFrame(df_profarea)
# Create the dataframe with id, name, labouring status of the specialization
df_specializations = []
for spec in specialization_id_name:
    df_specializations.extend(spec['specializations'])
df_specializations = pd.DataFrame(df_specializations)
df_specializations.columns = ['specialization_id', 'specialization_name', 'laboring']
df_specializations['profarea_id'] = df_specializations['specialization_id'].apply(
    lambda i: i.split('.')[0])
# Add profarea names to specializations
df_specializations = df_specializations.merge(df_profarea, how="outer")
# Sort by specialization id
df_specializations['specialization_id_float'] = df_specializations['specialization_id'].astype(float)
df_specializations = df_specializations.sort_values('specialization_id_float')
# Add number of vacancies
df_specializations = df_specializations.merge(cross_spec, how='left')
# Select only usefull columns
df_specializations = df_specializations.loc[:,['profarea_name', 'specialization_id',
                                               'specialization_name', 'laboring',
                                               'vac_number']]

### 3.7 Profarea preprocessing
cross_profarea = df_specializations.loc[:,['profarea_name', 'vac_number']].groupby(
    'profarea_name')['vac_number'].sum()
cross_profarea = pd.DataFrame(cross_profarea).reset_index()
cross_profarea.sort_values('vac_number', inplace=True, ascending=False)

# Save to pickle
os.chdir("/data")

# Dataset with vacancies from 2019 to 2020
df_hh.to_pickle("df_hh_2019_2020.obj")

# Dataset with vacancies from 2019 to 2020 only with skills
df_hh[df_hh['key_skills'].notna()].to_pickle("df_hh_2019_2020_skills.obj")
# df_hh = pd.read_pickle("df_hh_2019_2020_skills.obj")

# =============================================================================
# 4. Add text to vacancies 2019-2020 with skills
# =============================================================================

# Select ids 2019-2020 with skills
os.chdir("/data")
selected_ids = set(list(pd.read_pickle("df_hh_2019_2020_skills.obj")['id']))

# Set working directory
os.chdir('D:/hh_archive')

def batchPreprocessing_text(df_batch, selected_ids):

    # Select only id and description
    df_batch = df_batch.loc[:,['id','description']]
    # Remove missings
    df_batch = df_batch[df_batch['id'].notna()]
    # Select only unique vacancies 
    df_batch = df_batch.drop_duplicates('id')
    # Select only vacancies with skills
    df_batch = df_batch[df_batch['id'].apply(lambda x: x in selected_ids)]
    # Reset index
    df_batch.reset_index(drop=True, inplace=True)
    
    return df_batch


# Connect to SQLite
conn = sqlite3.connect("D:/hh_sql_db3/hh_text.db")
cur = conn.cursor()

# Read all file names from working directory 
files_in_dir = os.listdir("D:/hh_archive")
files_in_dir = list(filter(lambda file_name: file_name.startswith("df_"), files_in_dir))
table_names = []

for i in range(len(files_in_dir)):
    
    try: 
        # Load batch of 100,000 vacancies
        df_batch = pd.read_pickle(files_in_dir[i])
        
        if type(df_batch) != pd.DataFrame:
            # Remove missings
            df_batch = [e for e in df_batch if type(e) == dict]
            # Convert to dataframe
            df_batch = pd.DataFrame(df_batch)
        
        # Preprocessing
        df_batch = batchPreprocessing_text(df_batch, selected_ids)
        
        if len(df_batch) > 0:
            # Save to SQL
            df_batch.to_sql(files_in_dir[i], conn, if_exists="replace")
            table_names.append(files_in_dir[i])
        
        # Print stats
        print(i, files_in_dir[i], df_batch.shape)
        
    except:
        print(i, files_in_dir[i], " -------- ERROR")


### Merge batch tables into one dataframe
start_time = time.time()
# Rename first table
cur.execute("ALTER TABLE " + table_names[0] + " RENAME TO df_hh_text;")
# Insert other batches into the first one
union_all_part = "(" + " UNION ALL ".join(["SELECT * FROM " + file_name for file_name in table_names[1:]]) + ")"
cur.execute("INSERT INTO df_hh_text SELECT * FROM " + union_all_part)
conn.commit()
print("Merged in one table")
# Remove seperate batch tables
for file_name in table_names[1:]:
    cur.execute("DROP TABLE " + file_name)
    conn.commit()
    print(file_name)
# Close the connection
conn.close()
print("--- %s minutes ---" % ((time.time() - start_time) / 60))


# Intialize the connection to SQL
conn = sqlite3.connect("D:/hh_sql_db3/hh_text.db")
conn.text_factory = str
cur = conn.cursor()

# Load the dataset
start_time = time.time()
df_hh_text = pd.read_sql_query(("SELECT id, description FROM df_hh_text"), conn)
# Close the connection
conn.close()
print("--- %s minutes ---" % ((time.time() - start_time) / 60))


# Save to pickle
os.chdir("/data")
# Dataset with TEXT of vacancies from 2019 to 2020 with skills
df_hh_text.to_pickle("df_hh_text.obj")


# =============================================================================
# 5. Full dataset from 2014, only date, vacancy name, salary and key skills
# =============================================================================

# Intialize the connection to SQL
conn = sqlite3.connect("D:/hh_sql_db3/hh.db")
conn.text_factory = str
cur = conn.cursor()

# Load the dataset
start_time = time.time()
df_hh = pd.read_sql_query(("SELECT created_at_date, name, salary_mean, key_skills, id " 
                          "FROM df_hh"), conn)
# Close the connection
conn.close()
print("--- %s minutes ---" % ((time.time() - start_time) / 60))

### 5.1 Time preprocessing
# Select the data from 2014 onwards
df_hh = df_hh[df_hh['created_at_date'] >= "2014-01-01"]
# Remove vacancies after January 1, 2021
# df_hh = df_hh[df_hh['created_at_date'] < "2020-11-01"]
df_hh = df_hh[df_hh['created_at_date'] < "2021-01-01"]
# Replace datetime with just date in YYYY-MM-DD format
df_hh['created_at_date'] = df_hh['created_at_date'].apply(
         lambda d: d[:10])
# Create column with date in YYYY-MM format
df_hh['created_at_month'] = df_hh['created_at_date'].apply(lambda d: d[:-3])
# Convert to datetime
df_hh['created_at_month'] = pd.to_datetime(df_hh['created_at_month'])
df_hh['created_at_date'] = pd.to_datetime(df_hh['created_at_date'])
# Obtain the date in YYYY-WEEK format
df_hh['created_at_week'] = df_hh['created_at_date'].apply(
    lambda d: str(d.isocalendar()[0]) + "_" + str(d.isocalendar()[1]))

### 5.2 Salary preprocessing
# Remove outliers in the salary
df_hh['salary_mean'][df_hh['salary_mean'] <= np.nanquantile(df_hh['salary_mean'], 0.005)] = np.nan
df_hh['salary_mean'][df_hh['salary_mean'] >= np.nanquantile(df_hh['salary_mean'], 0.995)] = np.nan
# Create variable that indicates presence of the salary in the vacancy
df_hh['salary_present'] = df_hh['salary_mean'].notna().astype(int)

### 5.3 key_skills presence
df_hh["skills_present"] = df_hh['key_skills'].apply(lambda x: x != 'nan')
df_hh["skills_present"] = df_hh["skills_present"].astype(int)

# Save to pickle
os.chdir("/data")

# Dataset with vacancies from 2014 to 2020
df_hh.to_pickle("df_hh_2014_2020.obj")


# =============================================================================
# 6. Add educational level based on the description
# =============================================================================

# Dataset with TEXT of vacancies from 2019 to 2020 with skills
os.chdir("/data")
df_hh_text = pd.read_pickle("df_hh_text.obj")

# Remove </> from description and make lower case
df_hh_text['description_text'] = df_hh_text['description'].apply(
    lambda t: BeautifulSoup(t, 'html').get_text().lower())

# Extract indication of Higher and Specialized education
df_hh_text['higher_edu'] = df_hh_text['description_text'].apply(
    lambda s: int("высше" in s and " образовани" in s) )
df_hh_text['specialized_edu'] = df_hh_text['description_text'].apply(
    lambda s: int("средне" in s and " образовани" in s) )

# From dummies to one variable for educational level
df_hh_text['higher_edu'] = df_hh_text['higher_edu'].replace({1:2})
df_hh_text['edu_level'] = df_hh_text['specialized_edu'] + df_hh_text['higher_edu']
df_hh_text['edu_level'] = df_hh_text['edu_level'].replace({3:"specialized",
                                                           2:"higher",
                                                           1:"specialized",
                                                           0:"not_indicated"})
df_hh_text = df_hh_text.loc[:,['id','edu_level']]

# Add to dataset with skills
df_hh = pd.read_pickle("df_hh_2019_2020_skills.obj")
df_hh = df_hh.merge(df_hh_text, on='id')
# Save Dataset with vacancies from 2019 to 2020 and skills + edu_level
df_hh.to_pickle("df_hh_2019_2020_skills.obj")


# =============================================================================
# 7. Add large skill category
# =============================================================================

# Read from pickle
os.chdir("/data")
df_hh = pd.read_pickle("df_hh_2019_2020_skills.obj")
# Load tables with skills categorisation
skill_to_category = pd.read_excel("skill_to_category.xlsx")
category_to_large = pd.read_excel("category_to_large.xlsx")
# Create dict from skill to large category
skill_to_category = skill_to_category.merge(category_to_large)
skill_to_category = skill_to_category.set_index('skill')['large_category'].to_dict()
def skillToCategory(skill_name):
    if skill_name in skill_to_category:
        return skill_to_category[skill_name]
# Add large category for each skill
df_hh['skills_category'] = df_hh['key_skills'].apply(
    lambda v: list(set([skillToCategory(s) for s in v])))
# Remove vacancies with large categories of skills
df_hh = df_hh[df_hh['skills_category'].apply(lambda v: v != [None])]
df_hh['skills_category'] = df_hh['skills_category'].apply(lambda s: list(filter(None.__ne__, s)))
df_hh.reset_index(inplace=True, drop=True)

# Recode skill names
temp = pd.Series(np.concatenate(df_hh['skills_category'].values)).value_counts()
old_to_new_names = {'Социальные':'skill_social',
                     'Профессиональные, средняя квалификация':'skill_professional',
                     'Компьютерные специализированные':'skill_comp_spec',
                     'Компьютерные общие':'skill_comp_general',
                     'Административно-организационные':'skill_administrative',
                     'Клиенто-ориентированные':'skill_clientoriented',
                     'Личностные':'skill_personal',
                     'Управление людьми':'skill_hrm',
                     'Когнитивные':'skill_cognitive',
                     'Финансовые':'skill_financial',
                     'Работа с информацией в интернете':'skill_internetwork',
                     'Иностранный язык':'skill_foreignlang',
                     'Управление проектами':'skill_projectmanagment',
                     'Безопасность':'skill_security',
                     'Медицинские, медико-психологические':'skill_medical', 
                     'Юридические':'skill_law',
                     'Литературные':'skill_literature',
                     'Профессиональные, без квалификации':'skill_professional_noqual'}
df_hh['skills_category'] = df_hh['skills_category'].apply(lambda vec: [old_to_new_names[x] for x in vec])

# Save Dataset with vacancies from 2019 to 2020 and skills + edu_level + skills_category
df_hh.to_pickle("df_hh_2019_2020_skills.obj")


# =============================================================================
# 8. ISCO occupations
# =============================================================================

# Load the data
os.chdir("/data")
df_hh = pd.read_pickle("df_hh_2019_2020_skills.obj")
df_vacname_occupation = pd.read_pickle("vacname_occupation.obj")

### 9.1 Preprocessing of the vacancy 
stopwords = get_stop_words('ru')
def removeStopWords(string, stopwords=stopwords):

    result = [w for w in string.split() if w not in stopwords]
    result = " ".join(result)
    
    return result

vacnames_hh = pd.DataFrame(df_hh['name'].value_counts()).reset_index()
vacnames_hh.columns = ['name', 'freq']
vacnames_hh['name_preprocessed'] = vacnames_hh['name'].apply(
    lambda name: re.sub(r'\([^)]*\)', '', name))
vacnames_hh['name_preprocessed'] = vacnames_hh['name_preprocessed'].apply(
    lambda name: re.sub(r'"', '', name))
vacnames_hh['name_preprocessed'] = vacnames_hh['name_preprocessed'].apply(
    lambda name: re.sub(r'-', ' ', name))
vacnames_hh['name_preprocessed'] = vacnames_hh['name_preprocessed'].apply(
    lambda name: name.strip().lower())
vacnames_hh['name_preprocessed'] = vacnames_hh['name_preprocessed'].apply(
    lambda t: re.sub(r'[^A-Za-zА-Яа-я]+', ' ', t.lower()).strip())
vacnames_hh = vacnames_hh[vacnames_hh['name_preprocessed'] != '']
vacnames_hh['name_preprocessed'] = vacnames_hh['name_preprocessed'].apply(removeStopWords)
# Lemmatization
morph = pymorphy2.MorphAnalyzer()
vacnames_hh['name_preprocessed'] = vacnames_hh['name_preprocessed'].apply(
    lambda name: " ".join([morph.parse(t)[0].normal_form for t in name.split()]))

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


### 9.2 Filter out vacancy names with low SVM proba, OOV and NoClass
df_vacname_occupation['svm_filter'] = df_vacname_occupation['svm_proba'] > -0.5
df_vacname_occupation = df_vacname_occupation[df_vacname_occupation['svm_filter']]
print(df_vacname_occupation['freq'].sum())
df_vacname_occupation = df_vacname_occupation[~df_vacname_occupation['OOV']]
print(df_vacname_occupation['freq'].sum())
df_vacname_occupation = df_vacname_occupation[df_vacname_occupation['isco_3_name'] != "NoClass"]
print(df_vacname_occupation['freq'].sum())


### 9.3 Merge vacnames_hh with df_vacname_occupation
vacnames_hh = vacnames_hh.merge(
    df_vacname_occupation[['name_preprocessed', 'isco_3_code', 'isco_3_name']], on='name_preprocessed', how='left')

# Fix the most popular missings manually
vacnames_hh['isco_3_code'][vacnames_hh['name'] == "Комплектовщик"] = "933"
vacnames_hh['isco_3_name'][vacnames_hh['name'] == "Комплектовщик"] = "Неквалифицированные рабочие, занятые на транспорте и в хранении"

vacnames_hh['isco_3_code'][vacnames_hh['name'] == "Монтажник слаботочных систем"] = "821"
vacnames_hh['isco_3_name'][vacnames_hh['name'] == "Монтажник слаботочных систем"] = "Сборщики"

vacnames_hh['isco_3_code'][vacnames_hh['name'] == "Косметолог-эстетист"] = "514"
vacnames_hh['isco_3_name'][vacnames_hh['name'] == "Косметолог-эстетист"] = "Парикмахеры, косметологи и работники родственных занятий"

vacnames_hh['isco_3_code'][vacnames_hh['name'] == "Косметолог"] = "514"
vacnames_hh['isco_3_name'][vacnames_hh['name'] == "Косметолог"] = "Парикмахеры, косметологи и работники родственных занятий"

vacnames_hh['isco_3_code'][vacnames_hh['name'] == "Специалист по качеству"] = "214"
vacnames_hh['isco_3_name'][vacnames_hh['name'] == "Специалист по качеству"] = "Специалисты-профессионалы в области техники, исключая электротехников"

vacnames_hh['isco_3_code'][vacnames_hh['name'] == "Комплектовщик (Московская Область)"] = "933"
vacnames_hh['isco_3_name'][vacnames_hh['name'] == "Комплектовщик (Московская Область)"] = "Неквалифицированные рабочие, занятые на транспорте и в хранении"

vacnames_hh['isco_3_code'][vacnames_hh['name'] == "Документовед"] = "411"
vacnames_hh['isco_3_name'][vacnames_hh['name'] == "Документовед"] = "Офисные служащие общего профиля"

# Add ISCO to the vacancies dataset
dict_vacname_iscocode = dict(zip(
    vacnames_hh['name'].values,
    vacnames_hh['isco_3_code'].values))
dict_vacname_isconame = dict(zip(
    vacnames_hh['name'].values,
    vacnames_hh['isco_3_name'].values))
df_hh['isco_code_3'] = df_hh['name'].apply(
    lambda x: dict_vacname_iscocode[x] if x in dict_vacname_iscocode else np.nan)
df_hh['isco_name_3'] = df_hh['name'].apply(
    lambda x: dict_vacname_isconame[x] if x in dict_vacname_isconame else np.nan)

# Remove vacancies with missing ISCO
df_hh = df_hh[df_hh['isco_code_3'].notna()]
# Remove Товарные производители смешанной растениеводческой и животноводческой
df_hh = df_hh[df_hh['isco_name_3'] != "Товарные производители смешанной растениеводческой и животноводческой"]

# Save to pickle
os.chdir("/data")
df_hh.to_pickle("df_hh_2019_2020_skills_isco.obj")


# =============================================================================
# 9. Preprocessing for STATA
# =============================================================================

# Load from pickle
os.chdir("/data")
df_hh = pd.read_pickle("df_hh_2019_2020_skills_isco.obj")

# Create dummies for each skill category
dummy_categories = pd.get_dummies(
    df_hh['skills_category'].apply(pd.Series).stack()).sum(level=0)
df_hh = df_hh.join(dummy_categories)

# Recode profarea names
df_hh['profarea_name'] = df_hh['profarea_name'].apply(lambda v: eval(v))
list_profareas = np.unique(np.concatenate(df_hh['profarea_name'].values))
vec_dummies = []
for name in list_profareas:
    vec_dummies.append(df_hh['profarea_name'].apply(lambda vec: int(name in vec)))
    print(name)
temp_dummies = pd.DataFrame(vec_dummies).T
temp_dummies.columns = list_profareas
old_to_new_names = {
       'Автомобильный бизнес':'i_carbusiness',
       'Административный персонал':'i_admin',
       'Банки, инвестиции, лизинг':'i_investbanking',
       'Безопасность':'i_security',
       'Бухгалтерия, управленческий учет, финансы предприятия':'i_accounting',
       'Высший менеджмент':'i_topmanage',
       'Государственная служба, некоммерческие организации':'i_pubadmin',
       'Добыча сырья':'i_mining',
       'Домашний персонал':'i_housekeep',
       'Закупки':'i_perchasing',
       'Инсталляция и сервис':'i_instal',
       'Информационные технологии, интернет, телеком':'i_it',
       'Искусство, развлечения, масс-медиа':'i_art',
       'Консультирование':'i_consult',
       'Маркетинг, реклама, PR':'i_marketing',
       'Медицина, фармацевтика':'i_healthcare',
       'Наука, образование':'i_educ_research',
       'Начало карьеры, студенты':'i_startjob',
       'Продажи':'i_sales',
       'Производство, сельское хозяйство':'i_agricult',
       'Рабочий персонал':'i_workers',
       'Спортивные клубы, фитнес, салоны красоты':'i_sportfit',
       'Страхование':'i_insur',
       'Строительство, недвижимость':'i_construct',
       'Транспорт, логистика':'i_logistics',
       'Туризм, гостиницы, рестораны':'i_tourismhotels',
       'Управление персоналом, тренинги':'i_personmanage',
       'Юристы':'i_lawyers'
}
temp_dummies.columns = list(map(lambda x: old_to_new_names[x], temp_dummies.columns))
df_hh = df_hh.join(temp_dummies)

### Recode area_id to region names

# Load the list of areas id and corresponding names
areas_id_name = json.loads(requests.get("https://api.hh.ru/areas").text)
# Select only Russian areas
areas_id_name = areas_id_name[0]['areas'] 
# Append the coresponding regions to the areas
dict_areas_region = dict()
for region in areas_id_name:
    for area in region['areas']:
        dict_areas_region[area['id']] = region['name']
dict_areas_region['1'] = "Москва"
dict_areas_region['2'] = "Санкт-Петербург"
dict_areas_region['4632'] = "Москва"
df_hh['region'] = df_hh['area_id'].apply(lambda a: dict_areas_region[a])

# Translate region names from russian to english
import cyrtranslit
region_names_ru = list(df_hh['region'].unique())
region_names_eng = list(map(lambda x: cyrtranslit.to_latin(x, 'ru'), region_names_ru))
region_names_translate = dict(zip(region_names_ru, region_names_eng))
df_hh['region'] = df_hh['region'].apply(lambda x: region_names_translate[x])

# Save Dataset with vacancies from 2019 to 2020
df_hh.to_pickle("df_hh_2019_2020_stata.obj")

# =============================================================================
# 10. Save to STATA
# =============================================================================

# Load from pickle
os.chdir("/data")
df_hh = pd.read_pickle("df_hh_2019_2020_stata.obj")

# Save in reduced form database .csv format for Gimpleson
df_hh_stata = df_hh.drop(columns=['created_at_date', 'name', 'key_skills', 'area_id',
                         'profarea_name', 'specialization_id', 'employer_id', 
                         'id', 'created_at_week', 'skills_present', 'skills_category',
                         'isco_name_3'])
df_hh_stata = df_hh_stata.rename(columns={"salary_mean": "wage",
                                          "created_at_month": "posting_date",
                                          "salary_present":"wage_yes"})

# Save to dta 
df_hh_stata.to_stata("df_hh_isco.dta", version=None)


# =============================================================================
# 11. Testing ISCO categories
# =============================================================================

os.chdir("/data")
df_hh = pd.read_pickle("df_hh_2019_2020_skills_isco.obj")

### 2. Compare with ISCO stats
os.chdir("/data")
df_isco_stats = pd.read_excel("stats_isco.xlsx")
df_isco_stats = df_isco_stats[df_isco_stats['okz nas3'] != "Total"]
df_isco_stats = df_isco_stats[['okz nas3', 'Freq.']]
df_isco_stats.columns = ['isco_code_3', 'stat_freq']
df_isco_stats['isco_code_3'] = df_isco_stats['isco_code_3'].astype(str)

df_isco_comparision = df_hh['isco_code_3'].value_counts().reset_index()
df_isco_comparision.columns = ['isco_code_3', 'hh_freq']
isco_code_name = df_hh[['isco_code_3', 'isco_name_3']].drop_duplicates()
df_isco_comparision = df_isco_comparision.merge(isco_code_name)

df_isco_comparision = df_isco_comparision.merge(df_isco_stats, how="left")

df_isco_comparision['rate'] = df_isco_comparision['stat_freq'] / df_isco_comparision['hh_freq']
df_isco_comparision['stat_freq'] = df_isco_comparision['stat_freq'] / 1000
df_isco_comparision['hh_freq'] = df_isco_comparision['hh_freq'] / 1000
df_isco_comparision.corr()
df_isco_comparision = df_isco_comparision[['isco_code_3', 'isco_name_3', 'stat_freq', 'hh_freq', 'rate']]
df_isco_comparision.columns = ['isco_code_3', 'isco_name_3', 'stat_freq_1000', 'hh_freq_1000', 'stat_hh_rate']
df_isco_comparision.sort_values('isco_code_3', inplace=True)

### 3. Mean salary in each category
stats_salary = df_hh.groupby('isco_name_3')['salary_mean'].mean().reset_index()
stats_salary = stats_salary.merge(df_isco_comparision[['isco_code_3', 'isco_name_3']])
stats_salary = stats_salary[['isco_code_3', 'isco_name_3', 'salary_mean']]
stats_salary.sort_values('isco_code_3', inplace=True)

### 4. Cross: ISCO name vs. Education level
cross_isco_edulevel = pd.crosstab(df_hh['isco_name_3'], df_hh['edu_level'], normalize='index').reset_index()
cross_isco_edulevel = cross_isco_edulevel.merge(df_isco_comparision[['isco_code_3', 'isco_name_3']])
cross_isco_edulevel = pd.concat([cross_isco_edulevel.iloc[:,-1:],
                                 cross_isco_edulevel.iloc[:,:-1]], 1)
cross_isco_edulevel.sort_values('isco_code_3', inplace=True)

### 5. Cross: ISCO name vs. Schedule
cross_isco_schedule = pd.crosstab(df_hh['isco_name_3'], df_hh['schedule'], normalize='index').reset_index()
cross_isco_schedule = cross_isco_schedule.merge(df_isco_comparision[['isco_code_3', 'isco_name_3']])
cross_isco_schedule = pd.concat([cross_isco_schedule.iloc[:,-1:],
                                 cross_isco_schedule.iloc[:,:-1]], 1)
cross_isco_schedule.sort_values('isco_code_3', inplace=True)

### 6. Cross: ISCO name vs. Experience
cross_isco_experience = pd.crosstab(df_hh['isco_name_3'], df_hh['experience'], normalize='index').reset_index()
cross_isco_experience = cross_isco_experience.merge(df_isco_comparision[['isco_code_3', 'isco_name_3']])
cross_isco_experience = pd.concat([cross_isco_experience.iloc[:,-1:],
                                 cross_isco_experience.iloc[:,:-1]], 1)
cross_isco_experience.sort_values('isco_code_3', inplace=True)

### 7 Cross: ISCO names vs. Skills
isco_names = df_hh['isco_name_3'].unique()
cross_isco_skills = pd.DataFrame(pd.Series(np.concatenate(
        df_hh[df_hh['isco_name_3'] == isco_names[0]]['skills_category'].values)).value_counts(normalize=True))
for name in isco_names[1:]:
    temp = pd.DataFrame(pd.Series(np.concatenate(
        df_hh[df_hh['isco_name_3'] == name]['skills_category'].values)).value_counts(normalize=True))
    cross_isco_skills = cross_isco_skills.merge(temp, left_index=True, right_index=True, how='outer')
    print(name)
cross_isco_skills.columns = isco_names
cross_isco_skills = cross_isco_skills.T
cross_isco_skills = cross_isco_skills.fillna(0)
cross_isco_skills = cross_isco_skills.reset_index()
cross_isco_skills = cross_isco_skills.rename(columns={"index":"isco_name_3"})
cross_isco_skills = cross_isco_skills.merge(df_isco_comparision[['isco_code_3', 'isco_name_3']])
cross_isco_skills = pd.concat([cross_isco_skills.iloc[:,-1:],
                                 cross_isco_skills.iloc[:,:-1]], 1)
cross_isco_skills.sort_values('isco_code_3', inplace=True)

### 8. Save statistics to Excel

# Create a Pandas Excel writer using XlsxWriter as the engine.
writer = pd.ExcelWriter('ISCO_descriptive.xlsx', engine='xlsxwriter')

# Write each dataframe to a different worksheet.
df_isco_comparision.to_excel(writer, sheet_name='Rosstat_VS_HeadHunter', index=False)
stats_salary.to_excel(writer, sheet_name='Salary', index=False)
cross_isco_edulevel.to_excel(writer, sheet_name='Education', index=False)
cross_isco_schedule.to_excel(writer, sheet_name='Schedule', index=False)
cross_isco_experience.to_excel(writer, sheet_name='Experience', index=False)
cross_isco_skills.to_excel(writer, sheet_name='Skills', index=False)

# Close the Pandas Excel writer and output the Excel file.
writer.save()

