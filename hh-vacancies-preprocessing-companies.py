
# =============================================================================
#
# --- LIBRARIES
#
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
import fuzzyset
import requests
import urllib
from requests_html import HTML
from requests_html import HTMLSession
from natasha import (
    Segmenter,
    MorphVocab,
    NewsEmbedding,
    NewsMorphTagger,
    NewsSyntaxParser,
    NewsNERTagger,
    PER,
    NamesExtractor,
    Doc
)

# from boo import download, unpack, read_dataframe
pd.options.mode.chained_assignment = None 


# =============================================================================
#
# --- FUNCTIONS
#
# =============================================================================


def get_source(url):
    """Return the source code for the provided URL. 

    Args: 
        url (string): URL of the page to scrape.

    Returns:
        response (object): HTTP response object from requests_html. 
    """

    try:
        session = HTMLSession()
        response = session.get(url)
        return response

    except requests.exceptions.RequestException as e:
        print(e)


def get_results(query):
    
    query = urllib.parse.quote_plus(query)
    response = get_source("https://www.google.co.uk/search?q=" + query)
    
    return response


def parse_results(response):
    
    css_identifier_result = ".tF2Cxc"
    css_identifier_title = "h3"
    css_identifier_link = ".yuRUbf a"
    css_identifier_text = ".IsZvec"
    
    results = response.html.find(css_identifier_result)

    output = []
    
    for result in results:

        item = {
            'title': result.find(css_identifier_title, first=True).text,
            'link': result.find(css_identifier_link, first=True).attrs['href'],
            'text': result.find(css_identifier_text, first=True).text
        }
        
        output.append(item)
        
    return output
    

def google_search(query):
    response = get_results(query)
    return parse_results(response)


def errorStringFix(lst):
    
    if len(lst) > 0:
        if lst[0] == "ERROR":
            result = []
        else:
            result = lst
    else:
        result = lst
    
    return result


def getMostCommonItem(lst):
    return max(set(lst), key=lst.count)


def getINN(parsed_list, most_common=False):
    
    inn_list_title = list(map(lambda x: x['title'].split('ИНН '), parsed_list))
    inn_list_title = list(map(lambda x: x[1].split(' ')[0] if len(x) > 1 else 'none', inn_list_title))
    inn_list_title = list(map(lambda x: re.findall(r'\d+', x), inn_list_title))
    
    inn_list_text = list(map(lambda x: x['text'].split('ИНН '), parsed_list))
    inn_list_text = list(map(lambda x: x[1].split(' ')[0] if len(x) > 1 else 'none', inn_list_text))
    inn_list_text = list(map(lambda x: re.findall(r'\d+', x), inn_list_text))
    
    df_inn_list = pd.DataFrame({'inn_list_title':inn_list_title,
                                'inn_list_text':inn_list_text})
    
    inn_list = df_inn_list.apply(lambda x: list(set(x[0] + x[1])), 1).values
    
    try:
        
        if most_common:
            inn_list = list(filter(lambda x: len(x) > 0, inn_list))
            inn_list = list(map(lambda x: x[0], inn_list))
            inn = getMostCommonItem(inn_list)
        else:            
            inn = list(filter(lambda x: len(x) > 0, inn_list))[0][0]
            
    except:
        inn = np.nan
    
    
    return inn


def websiteNamesPreprocessingExportbase(website_list):
    
    if type(website_list) == list:
        website_list = list(map(lambda x: x.replace('http://', ''), website_list))
        website_list = list(map(lambda x: x.replace('www.', ''), website_list))
        website_list = list(map(lambda x: x.split('/')[0], website_list))
        website_list = set(website_list)
    else:
        website_list = np.nan
        
    return website_list


def namePresence(text):
    
    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_ner(ner_tagger)
    result = []
    for span in doc.spans:
        result.append(span.type == PER)
    return all([any(result), (len(doc.tokens) == 3)])


# =============================================================================
#
# --- 1. PREPROCESSING
#
# =============================================================================


###### --- 1.1 HeadHunter

### --- 1.1.1 HeadHunter, Vacancies

# Intialize the connection to SQL
conn = sqlite3.connect("D:/hh_sql_db3/hh.db")
conn.text_factory = str
cur = conn.cursor()

# Load the dataset
start_time = time.time()
df_hh = pd.read_sql_query(("SELECT created_at_date,  " 
                          "area_id, "
                          "employer_id, "
                          "address, "
                          "id " 
                          "FROM df_hh"), conn)
# Close the connection
conn.close()
print("--- %s minutes ---" % ((time.time() - start_time) / 60))

# TODO: Add coordinates or area_name

# Select the data from 2014 onwards
df_hh = df_hh[df_hh['created_at_date'] >= "2014-01-01"]
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
print("Date preprocessing is finished")

# Region preprocessing
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
print("Region preprocessing is finished")

# Add variable with the total number of vacancies per company
vac_total_n = df_hh['employer_id'].value_counts().reset_index()
vac_total_n.columns = ['employer_id', 'vac_total_n']
df_hh = df_hh.merge(vac_total_n, how='left')

# Save to pickle
os.chdir("/data")
df_hh.to_pickle("df_hh_2014_2020_companies.obj")
df_hh = pd.read_pickle("df_hh_2014_2020_companies.obj")

# Get unique list of companies ids
companies_ids = np.unique(df_hh['employer_id'][df_hh['employer_id'].notna()])

os.chdir("/data")
with open('companies_ids.pickle', 'wb') as f:
    pickle.dump(companies_ids, f)

### --- 1.1.2 HeadHunter, Companies

os.chdir("/data/companies_info")
files_in_dir = os.listdir()

df_companies_hh = []
for file_name in files_in_dir:
    temp = pd.read_pickle(file_name)
    df_companies_hh = df_companies_hh + temp
    print(file_name)

# Transform to dataframe
df_companies_hh = pd.DataFrame(df_companies_hh)
# Remove Not Found companies
df_companies_hh = df_companies_hh[df_companies_hh['description'] != 'Not Found']
# Extract id and industy name
df_companies_hh['industries'] = df_companies_hh['industries'].apply(
    lambda v: [e['id'] + '_' + e['name'] for e in v] if len(v)>0 else np.nan)

# Extract area name
df_companies_hh['area_id'] = df_companies_hh['area'].apply(lambda x: x['id'])
df_companies_hh['area_name'] = df_companies_hh['area'].apply(lambda x: x['name'])
df_companies_hh.drop('area', 1, inplace=True)
df_companies_hh['description'].notna().sum()
df_companies_hh['industries'].notna().sum()

# Add total number of vacancies
os.chdir("/data")
df_hh = pd.read_pickle("df_hh_2014_2020_companies.obj")
df_hh = df_hh[df_hh.created_at_date > "2019-01-01"] # <------------------ Select only 2019-2020
df_hh.drop_duplicates("employer_id", inplace=True)
df_companies_hh = df_companies_hh.merge(df_hh[['employer_id', 'vac_total_n']],
                                  how='left', left_on='id', right_on='employer_id')
df_companies_hh = df_companies_hh[df_companies_hh['vac_total_n'].notna()]
df_companies_hh.drop('employer_id', 1, inplace=True)

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
df_companies_hh['region'] = df_companies_hh['area_id'].apply(
    lambda a: dict_areas_region[a] if a in dict_areas_region else np.nan)
df_companies_hh = df_companies_hh[df_companies_hh['region'].notna()]

# Remove Agencies and Private recruters
df_companies_hh = df_companies_hh[df_companies_hh['type'].apply(lambda t: t not in {"agency", "private_recruiter"})]

# Preprocessing of companies site_url HH
df_companies_hh['site_url'] = df_companies_hh['site_url'].apply(lambda url: url.replace('http://', ''))
df_companies_hh['site_url'] = df_companies_hh['site_url'].apply(lambda url: url.replace('http:', ''))
df_companies_hh['site_url'] = df_companies_hh['site_url'].apply(lambda url: url.replace('https:', ''))
df_companies_hh['site_url'] = df_companies_hh['site_url'].apply(lambda url: url.replace('vk.com', ''))
df_companies_hh['site_url'] = df_companies_hh['site_url'].apply(lambda url: url.replace('instagram.com', ''))
df_companies_hh['site_url'] = df_companies_hh['site_url'].apply(lambda url: url.replace('Instagram.com', ''))
df_companies_hh['site_url'] = df_companies_hh['site_url'].apply(lambda url: url.replace('www.', ''))
df_companies_hh['site_url'] = df_companies_hh['site_url'].apply(lambda url: url.split('/')[0])
df_companies_hh['site_url'] = df_companies_hh['site_url'].replace({'':np.nan})

# Sort by the number of total vacancies
df_companies_hh = df_companies_hh.sort_values('vac_total_n', ascending=False)

# Save to pickle
os.chdir("/data")
df_companies_hh.to_pickle("df_companies_hh.obj")


###### --- 1.2 Export-base

os.chdir("/data/export_base")

# Load the different parts
df_companies_exportbase_1 = pd.read_excel("export_base_1.xlsx", dtype={"V4":str, "V5":str})
print(1)
df_companies_exportbase_2 = pd.read_excel("export_base_2.xlsx", dtype={"V4":str, "V5":str})
print(2)
df_companies_exportbase_3 = pd.read_excel("export_base_3.xlsx", dtype={"V4":str, "V5":str})
print(3)
df_companies_exportbase_4 = pd.read_excel("export_base_4.xlsx", dtype={"V4":str, "V5":str})
print(4)
df_companies_exportbase_5 = pd.read_excel("export_base_5.xlsx", dtype={"V4":str, "V5":str})
print(5)
df_companies_exportbase_6 = pd.read_excel("export_base_6.xlsx", dtype={"V4":str, "V5":str})
print(6)
df_companies_exportbase_7 = pd.read_excel("export_base_7.xlsx", dtype={"V4":str, "V5":str})
print(7)
df_companies_exportbase_8 = pd.read_excel("export_base_8.xlsx", dtype={"V4":str, "V5":str})
print(8)
df_companies_exportbase_9 = pd.read_excel("export_base_9.xlsx", dtype={"V4":str, "V5":str})
print(9)

# Concatenate parts together
df_companies_exportbase = pd.concat([df_companies_exportbase_1,
                                     df_companies_exportbase_2,
                                     df_companies_exportbase_3,
                                     df_companies_exportbase_4,
                                     df_companies_exportbase_5,
                                     df_companies_exportbase_6,
                                     df_companies_exportbase_7,
                                     df_companies_exportbase_8,
                                     df_companies_exportbase_9])
df_companies_exportbase.reset_index(inplace=True, drop=True)

# Rename columns
df_companies_exportbase.columns = ['name',
                                   'official_name',
                                   'website',
                                   'inn',
                                   'ogrn',
                                   'n_staff',
                                   'revenue',
                                   'registration_year',
                                   'okved_code', 'okved_name', 'city', 'region', 'federal_district',
                                   'category', 'subcategory', 'has_branches', 'coordinates',
                                   'time_zone', 'unnamed_column_1',
                                   'revenue_change',
                                   'revenue_2020', 'revenue_2019', 'revenue_2018', 'revenue_2017',
                                   'unnamed_column_2',
                                   'profit_change',
                                   'profit_2020', 'profit_2019', 'profit_2018', 'profit_2017',
                                   'operating_status',
                                   'unnamed_column_3']

# Drop redundant columns
df_companies_exportbase.drop(
    columns = ['revenue', 'time_zone', 'unnamed_column_1',
               'revenue_change', 'unnamed_column_2', 'profit_change', 'unnamed_column_3'],
    inplace=True)

# Remove technical rows
df_companies_exportbase = df_companies_exportbase[df_companies_exportbase['inn'].notna()]

# Varibles Preprocessing
df_companies_exportbase['registration_year'] = df_companies_exportbase['registration_year'].replace({'Н/Д':np.nan})
df_companies_exportbase['has_branches'] = df_companies_exportbase['has_branches'].replace({'X':True, np.nan:False})
df_companies_exportbase['revenue_2020'] = df_companies_exportbase['revenue_2020'].replace({'-':np.nan}).astype(np.float64)
df_companies_exportbase['revenue_2019'] = df_companies_exportbase['revenue_2019'].replace({'-':np.nan}).astype(np.float64)
df_companies_exportbase['revenue_2018'] = df_companies_exportbase['revenue_2018'].replace({'-':np.nan}).astype(np.float64)
df_companies_exportbase['revenue_2017'] = df_companies_exportbase['revenue_2017'].replace({'-':np.nan}).astype(np.float64)
df_companies_exportbase['profit_2020'] = df_companies_exportbase['profit_2020'].replace({'-':np.nan}).astype(np.float64)
df_companies_exportbase['profit_2019'] = df_companies_exportbase['profit_2019'].replace({'-':np.nan}).astype(np.float64)
df_companies_exportbase['profit_2018'] = df_companies_exportbase['profit_2018'].replace({'-':np.nan}).astype(np.float64)
df_companies_exportbase['profit_2017'] = df_companies_exportbase['profit_2017'].replace({'-':np.nan}).astype(np.float64)

### --- Company name preprocessing

# Short name
df_companies_exportbase['name_cleaned'] = df_companies_exportbase['name'].apply(lambda n: n.split(',')[0])
df_companies_exportbase['name_cleaned'] = df_companies_exportbase['name_cleaned'].apply(lambda n: n.lower())

# Official name
df_companies_exportbase['official_name_cleaned'] = df_companies_exportbase['official_name'].fillna('')
df_companies_exportbase['official_name_cleaned'] = df_companies_exportbase['official_name_cleaned'].apply(
    lambda n: n.replace('АНО ', '') if n.startswith('АНО ') else n)
df_companies_exportbase['official_name_cleaned'] = df_companies_exportbase['official_name_cleaned'].apply(
    lambda n: n.replace('АО ', '') if n.startswith('АО ') else n)
df_companies_exportbase['official_name_cleaned'] = df_companies_exportbase['official_name_cleaned'].apply(
    lambda n: n.replace('ИП ', '') if n.startswith('ИП ') else n)
df_companies_exportbase['official_name_cleaned'] = df_companies_exportbase['official_name_cleaned'].apply(
    lambda n: n.replace('ООО ', '') if n.startswith('ООО ') else n)
df_companies_exportbase['official_name_cleaned'] = df_companies_exportbase['official_name_cleaned'].apply(
    lambda n: n.replace('ЗАО ', '') if n.startswith('ЗАО ') else n)
df_companies_exportbase['official_name_cleaned'] = df_companies_exportbase['official_name_cleaned'].apply(
    lambda n: n.replace('"', ''))
df_companies_exportbase['official_name_cleaned'] = df_companies_exportbase['official_name_cleaned'].apply(lambda n: n.lower())


### --- Website preprocessing

# Additional preprocessing of site_urls, split by |
df_companies_exportbase['website_cleaned'] = df_companies_exportbase['website'].apply(
    lambda url: url.split(' | ') if type(url) == str else np.nan)
df_companies_exportbase['website_cleaned'] = df_companies_exportbase['website_cleaned'].apply(websiteNamesPreprocessingExportbase)


### --- Region preprocessing
df_companies_exportbase['region'][df_companies_exportbase['city'] == 'Москва'] = 'Москва'
df_companies_exportbase['region'][df_companies_exportbase['city'] == 'Санкт-Петербург'] = 'Санкт-Петербург'

dict_hh_exportbase_region_names = pd.read_excel("hh_exportbase_region_names.xlsx")
dict_hh_exportbase_region_names = dict(zip(
    dict_hh_exportbase_region_names['exportbase_region_name'].values,
    dict_hh_exportbase_region_names['hh_region_name'].values
    ))
df_companies_exportbase['region_cleaned'] = df_companies_exportbase['region'].replace(
    dict_hh_exportbase_region_names)


### --- Number of Staff preprocessing
df_companies_exportbase['n_staff_cleaned'] = df_companies_exportbase['n_staff'].apply(lambda x: x.replace(" ", "") if type(x) == str else np.nan)
df_companies_exportbase['n_staff_cleaned_numeric'] = df_companies_exportbase['n_staff_cleaned'].replace({"<15":"10", "16-100":"28", "101-250":"175"})
df_companies_exportbase['n_staff_cleaned_numeric'] = df_companies_exportbase['n_staff_cleaned_numeric'].astype(np.float64)
df_companies_exportbase['n_staff_cleaned_interval'] = pd.cut(df_companies_exportbase['n_staff_cleaned_numeric'],
                                                             [-1, 15, 100, 250, np.inf],
                                                             labels=['<15', '16-100', '101-250', '>251'])

# Save Wide version to pickle
os.chdir("/data")
df_companies_exportbase.to_pickle("df_companies_exportbase_wide.obj")
df_companies_exportbase_wide = pd.read_pickle("df_companies_exportbase_wide.obj")

### --- Middle version: Create middle version with unique INN + Region
df_companies_exportbase_middle = df_companies_exportbase_wide[df_companies_exportbase_wide['inn'] != '0']
df_companies_exportbase_middle['exportbase_id'] = df_companies_exportbase_middle['inn'] + "_" + df_companies_exportbase_middle['region_cleaned']
df_companies_exportbase_middle.drop_duplicates('exportbase_id', inplace=True)
os.chdir("/data")
df_companies_exportbase_middle.to_pickle("df_companies_exportbase_middle.obj")

### --- Short version: Create short version with unique INN
df_companies_exportbase_middle = pd.read_pickle('df_companies_exportbase_middle.obj')
df_companies_exportbase_short = df_companies_exportbase_middle.drop(
    ['city', 'region', 'federal_district', 'category',
     'subcategory', 'has_branches', 'operating_status',
     'exportbase_id'], 1)
df_companies_exportbase_short['website_cleaned'] = df_companies_exportbase_short['website_cleaned'].apply(
    lambda x: list(x)[0] if type(x)==set else np.nan)

df_companies_exportbase_short_1 = df_companies_exportbase_short[['inn', 'ogrn', 
                               'registration_year', 'okved_code', 'okved_name', 'revenue_2020',
                               'revenue_2019', 'revenue_2018', 'revenue_2017', 'profit_2020',
                               'profit_2019', 'profit_2018', 'profit_2017',
                               'n_staff', 'n_staff_cleaned', 'n_staff_cleaned_numeric',
                               'n_staff_cleaned_interval']]

df_companies_exportbase_short_1.drop_duplicates('inn', inplace=True)
df_companies_exportbase_short_2 = df_companies_exportbase_short[['inn', 'name', 'official_name', 'website',
                                   'name_cleaned','official_name_cleaned', 'website_cleaned', 'coordinates',
                                   'region_cleaned']].groupby(by="inn", as_index=False).agg(set)
df_companies_exportbase_short = df_companies_exportbase_short_1.merge(
    df_companies_exportbase_short_2, on='inn')

# OKVED preprocessing
# df_companies_exportbase_short = pd.read_pickle("df_companies_exportbase_short.obj")


def getOkvedSubclass_2(okved_code):
    
    okved_code_subclass = np.nan
    if type(okved_code) == str:
        if len(okved_code) >= 2:
            okved_code_subclass = okved_code[:2]
    
    return okved_code_subclass


def getOkvedSubclass_3(okved_code):
    
    okved_code_subclass = np.nan
    if type(okved_code) == str:
        if len(okved_code) >= 4:
            okved_code_subclass = okved_code[:4]
    
    return okved_code_subclass


def getOkvedSubclass_4(okved_code):
    
    okved_code_subclass = np.nan
    if type(okved_code) == str:
        if len(okved_code) >= 5:
            okved_code_subclass = okved_code[:5]
    
    return okved_code_subclass


# Extract OKVED codes: 3rd and 4th level
df_companies_exportbase_short['okved_code_2'] = df_companies_exportbase_short['okved_code'].apply(getOkvedSubclass_2)
df_companies_exportbase_short['okved_code_3'] = df_companies_exportbase_short['okved_code'].apply(getOkvedSubclass_3)
df_companies_exportbase_short['okved_code_4'] = df_companies_exportbase_short['okved_code'].apply(getOkvedSubclass_4)

# Add OKVED names
dict_okved_code_name = pd.read_excel("okved_code_name.xlsx")
dict_okved_code_name = dict_okved_code_name[dict_okved_code_name['okved_code'].notna()]
dict_okved_code_name = dict_okved_code_name[~dict_okved_code_name['okved_code'].apply(lambda x: x.startswith("РАЗДЕЛ"))]
dict_okved_code_name['okved_code'] = dict_okved_code_name['okved_code'].apply(lambda x: x.strip())
dict_okved_code_name['okved_name'] = dict_okved_code_name['okved_name'].apply(lambda x: x.strip())
dict_okved_code_name = dict(zip(
    dict_okved_code_name['okved_code'].values,
    dict_okved_code_name['okved_name'].values
    ))

df_companies_exportbase_short['okved_name_2'] = df_companies_exportbase_short['okved_code_2'].apply(
    lambda x: dict_okved_code_name[x] if x in dict_okved_code_name else np.nan)
df_companies_exportbase_short['okved_name_3'] = df_companies_exportbase_short['okved_code_3'].apply(
    lambda x: dict_okved_code_name[x] if x in dict_okved_code_name else np.nan)
df_companies_exportbase_short['okved_name_4'] = df_companies_exportbase_short['okved_code_4'].apply(
    lambda x: dict_okved_code_name[x] if x in dict_okved_code_name else np.nan)

# Coordinates preprocessing
df_companies_exportbase_short['coordinates'] = df_companies_exportbase_short['coordinates'].apply(lambda x: 
                                                   set(', '.join(list(x)).split(', ')))
df_companies_exportbase_short['coordinates'] = df_companies_exportbase_short['coordinates'].apply(
    lambda x: np.nan if list(x)[0]=='0.000 0.000' else x)

# Save to pickle
os.chdir("/data")
df_companies_exportbase_short.to_pickle("df_companies_exportbase_short.obj")


# =============================================================================
#
# --- 2. PARSING
#
# =============================================================================

###### --- 2.1 Parsing by name + region + inn

os.chdir("/data")
df_companies_hh = pd.read_pickle("df_companies_hh.obj")

os.chdir("/data/parsed_companies")
parsed_results = []
for i in range(len(df_companies_hh)):
    
    search_query = df_companies_hh['name'].iloc[i] + " " + df_companies_hh['region'].iloc[i] + " ИНН"
    
    try:
        search_results = google_search(search_query)
    except:
        search_results = ["ERROR"]
        print("-------- ERROR")
    if len(search_results) == 0:
        print("-------- EMPTY LIST")
    parsed_results.append(search_results)
    
    if i % 1e4 == 0:
        pd.Series(parsed_results).to_pickle("parsed_companies_" + str(i) + ".obj")
        print('save')
    
    time.sleep(0.3 + np.random.beta(1,1))
    print(i)

pd.Series(parsed_results).to_pickle("parsed_companies_full.obj")


###### --- 2.2 Additional parsing

os.chdir("/data/parsed_companies")
parsed_results = pd.read_pickle("parsed_companies_full.obj")
df_companies_hh['parsed_results'] = parsed_results.values

# Extract companies with ИП names from the parentheses and repeat parsing
df_companies_hh_failed = df_companies_hh[df_companies_hh['parsed_results'].apply(len) == 0]
df_companies_hh = df_companies_hh[df_companies_hh['parsed_results'].apply(len) != 0]
df_companies_hh_failed['name'] = df_companies_hh_failed['name'].apply(
    lambda name: name.split(" (ИП ")[1][:-1] if len(name.split(" (ИП ")) > 1 else name)

parsed_results_failed = []
for i in range(len(df_companies_hh_failed)):
    
    search_query = df_companies_hh_failed['name'].iloc[i] + " " + df_companies_hh_failed['region'].iloc[i] + " ИНН"
    
    try:
        search_results = google_search(search_query)
    except:
        search_results = ["ERROR"]
        print("ERROR")
    parsed_results_failed.append(search_results)
    
    time.sleep(0.3 + np.random.beta(1,1))
    print(i)

pd.Series(parsed_results_failed).to_pickle('parsed_results_failed.obj')
parsed_results_failed = pd.read_pickle("parsed_results_failed.obj")
df_companies_hh_failed['parsed_results'] = parsed_results_failed.values
df_companies_hh = pd.concat([df_companies_hh, df_companies_hh_failed])

###### --- 2.3 Preprocessing of the parsed results

# Prerpocessing of the parsed results
df_companies_hh['parsed_results_post'] = df_companies_hh['parsed_results'].apply(errorStringFix)
df_companies_hh['parsed_results_post'] = df_companies_hh['parsed_results_post'].apply(
    lambda lst: list(filter(lambda x: ("ИНН" in x['text'] or "ИНН" in x['title']), lst)) if len(lst) > 0 else lst)

# Extract INN in the
df_companies_hh['inn_first'] = df_companies_hh['parsed_results_post'].apply(getINN, most_common=False)
df_companies_hh['inn_mostcommon'] = df_companies_hh['parsed_results_post'].apply(getINN, most_common=True)
df_companies_hh['match'] = df_companies_hh['inn_first'] == df_companies_hh['inn_mostcommon']
df_companies_hh['inn_present'] = df_companies_hh['inn_first'].notna()

###### --- 2.4 Manual correction of the first 1,000 companies by the total number of vacancies

df_companies_hh_head = df_companies_hh.head(1000)
df_companies_hh_head = df_companies_hh_head[['name', 'region', 'vac_total_n', 'inn_present', 'inn_first', 'inn_mostcommon', 'match']]
df_companies_hh_head['match'] = df_companies_hh_head['match'].astype(int)
df_companies_hh_head['inn_present'] = df_companies_hh_head['inn_present'].astype(int)

os.chdir("/data")
# df_companies_hh_head.to_excel("df_companies_hh_head.xlsx")

# Load coded companies
os.chdir("/data")
df_companies_hh_head = pd.read_excel("df_companies_hh_head.xlsx",
                                     dtype=str)
df_companies_hh_head['inn_correct'] = df_companies_hh_head['inn_correct'].apply(lambda x: x.strip() if type(x)==str else np.nan)

# Preprocessing
df_companies_hh_head['inn_correct'] = df_companies_hh_head['inn_correct'].apply(lambda inn: inn.strip() if type(inn)==str else np.nan)
df_companies_hh_head['full_match'] = df_companies_hh_head['full_match'].replace({"True":True, "False":False})
df_companies_hh_head_fill = df_companies_hh_head[df_companies_hh_head['inn_correct'].notna()]
df_companies_hh_head_na = df_companies_hh_head[df_companies_hh_head['inn_correct'].isna()]
df_companies_hh_head_na = df_companies_hh_head_na[df_companies_hh_head_na['full_match'].values]
df_companies_hh_head_na['inn_correct'] = df_companies_hh_head_na[['inn_first', 'inn_mostcommon', 'inn_first_url', 'inn_mostcommon_url']].apply(
    lambda lst: [e for e in lst if type(e) == str][0], 1).values

# Combine 
df_companies_hh_head = pd.concat([df_companies_hh_head_na, df_companies_hh_head_fill])

# Add to the main dataset
df_companies_hh_1 = df_companies_hh[:1000]
df_companies_hh_2 = df_companies_hh[1000:]
df_companies_hh_1 = df_companies_hh_1.merge(df_companies_hh_head[['id', 'inn_correct']])
df_companies_hh_1 = df_companies_hh_1[df_companies_hh_1['inn_correct'].notna()]
df_companies_hh_1['inn_first'] = df_companies_hh_1['inn_correct'].values
df_companies_hh_1['inn_mostcommon'] = df_companies_hh_1['inn_correct'].values
df_companies_hh = pd.concat([df_companies_hh_1, df_companies_hh_2])
df_companies_hh = df_companies_hh.drop('inn_correct', 1)
df_companies_hh = df_companies_hh.rename(columns={"match":"inn_first_mostcommon_match"})

# Save to pickle
os.chdir("/data")
df_companies_hh.to_pickle("df_companies_hh.obj")


# =============================================================================
#
# --- 3. MATCHING
#
# =============================================================================

# Load hh and export-base
os.chdir("/data")
df_companies_hh = pd.read_pickle("df_companies_hh.obj")
df_companies_exportbase_short = pd.read_pickle("df_companies_exportbase_short.obj")
# Add prefix to column names
df_companies_hh.columns = ["hh_" + x for x in df_companies_hh.columns]
df_companies_exportbase_short.columns = ["exp_" + x for x in df_companies_exportbase_short.columns]

###### --- 3.1 Website-based matching

# Only for Unique sites value_counts == 1
# Select for HH
unique_sites_hh = df_companies_hh['hh_site_url'].value_counts() 
unique_sites_hh = set(unique_sites_hh[unique_sites_hh == 1].index)
df_companies_hh_unique_url = df_companies_hh[df_companies_hh['hh_site_url'].apply(lambda s: s in unique_sites_hh)]
# Select for Export-Base
df_companies_exportbase_short['exp_website_cleaned'] = df_companies_exportbase_short['exp_website_cleaned'].apply(
    lambda x: list(x)[0] if type(x)==set else np.nan) # TODO: Rewrite in a better way
unique_sites_exportbase = df_companies_exportbase_short['exp_website_cleaned'].value_counts() 
unique_sites_exportbase = set(unique_sites_exportbase[unique_sites_exportbase == 1].index)
df_companies_exportbase_short_unique_url = df_companies_exportbase_short[df_companies_exportbase_short['exp_website_cleaned'].apply(
    lambda s: s in unique_sites_exportbase)]

# Merge hh and exportbase based on the common website
df_companies_hh_unique_url['hh_site_url'] = df_companies_hh_unique_url['hh_site_url'].apply(lambda x: x.lower())
df_companies_unique_url = df_companies_hh_unique_url[['hh_id', 'hh_site_url']].merge(
    df_companies_exportbase_short_unique_url[['exp_inn', 'exp_website_cleaned']],
    how='inner', left_on='hh_site_url', right_on='exp_website_cleaned')

# Add INN obtain by this method to the main dataset
dict_id_inn = dict(zip(
    df_companies_unique_url['hh_id'].values,
    df_companies_unique_url['exp_inn'].values
    ))
df_companies_hh['inn_by_website'] = df_companies_hh['hh_id'].apply(lambda x: dict_id_inn[x] if x in dict_id_inn else np.nan)


###### --- 3.2 Name-based matching

# Read from pickle
os.chdir("/data")
df_companies_exportbase_middle = pd.read_pickle("df_companies_exportbase_middle.obj")

# Preprocessing of HH companies
df_companies_hh['hh_region'] = df_companies_hh['hh_region'].replace({"Ненецкий АО":"Архангельская область"})
df_companies_hh['hh_name_cleaned'] = df_companies_hh['hh_name'].apply(lambda x: x.lower())
df_companies_hh['hh_name_cleaned'] = df_companies_hh['hh_name_cleaned'].apply(lambda n: n.split(',')[0])
df_companies_hh['hh_name_cleaned'] = df_companies_hh['hh_name_cleaned'].apply(lambda n: re.sub("[^a-zA-Zа-яА-Я0-9]+", "", n))

# --- Split company by region
df_hh = pd.read_pickle("df_hh_2014_2020_companies.obj")
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
# Add additional names
additional_ids = set()
df_hh['region'] = df_hh['area_id'].apply(
    lambda a: dict_areas_region[a] if a in dict_areas_region else additional_ids.add(a))
df_hh['region'] = df_hh['region'].replace({"Ненецкий АО":"Архангельская область"})
dict_employer_vac_regions = df_hh.groupby('employer_id')['region'].apply(set)
dict_employer_vac_regions = dict(zip(
    dict_employer_vac_regions.index,
    dict_employer_vac_regions.values))
df_companies_hh['hh_regions_of_vacancies'] = df_companies_hh['hh_id'].apply(lambda x: dict_employer_vac_regions[x])
df_companies_hh['hh_regions_of_vacancies'] = [df_companies_hh['hh_regions_of_vacancies'].iloc[i].union(
    set([df_companies_hh['hh_region'].iloc[i]])) for i in range(len(df_companies_hh))]

### Additional prepocessing of names
# Remove non Alphabet and Digits symbols
df_companies_exportbase_middle['name_cleaned'] = df_companies_exportbase_middle['name_cleaned'].apply(
    lambda n: re.sub("[^a-zA-Zа-яА-Я0-9]+", "", n))
df_companies_exportbase_middle['official_name_cleaned'] = df_companies_exportbase_middle['official_name_cleaned'].apply(
    lambda n: re.sub("[^a-zA-Zа-яА-Я0-9]+", "", n))

# --- Match between companies for each regions, NAME

region_names = list(df_companies_hh['hh_region'].unique())
df_names_similarity = pd.DataFrame()
for name in region_names:
    
    # Subset of the one region
    df_companies_hh_region =  df_companies_hh[df_companies_hh['hh_region'] == name][['hh_id', 'hh_name_cleaned', 'hh_region']]
    df_companies_exportbase_region = df_companies_exportbase_middle[df_companies_exportbase_middle['region_cleaned'] == name][['inn', 'name_cleaned', 'official_name_cleaned', 'region_cleaned']]
    
    df_companies_hh_region_1 = df_companies_hh_region.merge(df_companies_exportbase_region,
                                                            how='inner', left_on='hh_name_cleaned', right_on='name_cleaned')
    temp_1 = df_companies_hh_region_1['name_cleaned'].value_counts()
    nonunique_names_1 = set(temp_1[temp_1 > 1].index)
    df_companies_hh_region_1 = df_companies_hh_region_1[df_companies_hh_region_1['name_cleaned'].apply(lambda x: x not in nonunique_names_1)]
        
    df_companies_hh_region_2 = df_companies_hh_region.merge(df_companies_exportbase_region,
                                                            how='inner', left_on='hh_name_cleaned', right_on='official_name_cleaned')
    temp_2 = df_companies_hh_region_2['official_name_cleaned'].value_counts()
    nonunique_names_2 = set(temp_2[temp_2 > 1].index)
    df_companies_hh_region_2 = df_companies_hh_region_2[df_companies_hh_region_2['official_name_cleaned'].apply(lambda x: x not in nonunique_names_2)]
    
    df_names_similarity_region = pd.concat([df_companies_hh_region_1, df_companies_hh_region_2])
    df_names_similarity_region = df_names_similarity_region.drop_duplicates('hh_id')
    
    # Append region data to the full database
    df_names_similarity = df_names_similarity.append(df_names_similarity_region)
    print(name)

df_names_similarity.to_pickle("df_names_similarity.obj")
df_names_similarity = pd.read_pickle("df_names_similarity.obj")

# Add INN obtain by this method to the main dataset
dict_id_inn = dict(zip(
    df_names_similarity['hh_id'].values,
    df_names_similarity['inn'].values
    ))
df_companies_hh['inn_by_names'] = df_companies_hh['hh_id'].apply(lambda x: dict_id_inn[x] if x in dict_id_inn else np.nan)


### Remove ИПs
segmenter = Segmenter()
morph_vocab = MorphVocab()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
syntax_parser = NewsSyntaxParser(emb)
ner_tagger = NewsNERTagger(emb)
names_extractor = NamesExtractor(morph_vocab)
df_companies_hh['ip'] = df_companies_hh['hh_name'].apply(namePresence)

# Save to pickle 
df_companies_hh.to_pickle("df_companies_hh.obj")

###### --- 3.3 Combine HH and Export-base

# Load hh and export-base
os.chdir("/data")
df_companies_hh = pd.read_pickle("df_companies_hh.obj")
df_companies_exportbase_short = pd.read_pickle("df_companies_exportbase_short.obj")
df_companies_exportbase_short.columns = ["exp_" + x for x in df_companies_exportbase_short.columns]

def combineINN(row):
    
    if row['hh_id'] in manually_checked_ids:
        result_inn = row['hh_inn_first']
        
    elif type(row['inn_by_website']) == str:
        result_inn = row['inn_by_website']
    elif type(row['inn_by_names']) == str:
        result_inn = row['inn_by_names']
    else:
        result_inn = row['hh_inn_first']
    
    return result_inn  


# Combine INN by website match, INN by parsing, INN by name and manually found INNs
manually_checked_ids = set(df_companies_hh['hh_id'].head(965).values)
df_companies_hh['hh_inn_combined'] = df_companies_hh[['hh_id', 'hh_inn_first', 'inn_by_website', 'inn_by_names']].apply(
    combineINN, 1)
# Change IP for one company
df_companies_hh['hh_inn_combined'][df_companies_hh['hh_id'] == "49466"] = "5033001802"
df_companies = df_companies_hh.merge(
    df_companies_exportbase_short, left_on='hh_inn_combined', right_on='exp_inn', how='left')
df_companies['exp_hh_match'] = df_companies['exp_inn'].notna()

# Save to pickle
df_companies.to_pickle('df_companies.obj')

# =============================================================================
#
# --- 4. EDA
#
# =============================================================================

# Read from pickle
os.chdir("/data")
df_companies = pd.read_pickle('df_companies.obj')
df_companies_exportbase_short = pd.read_pickle("df_companies_exportbase_short.obj")
df_companies_exportbase_short.columns = ["exp_" + x for x in df_companies_exportbase_short.columns]

# Stats
print(df_companies['exp_hh_match'].sum() / len(df_companies))
print(df_companies.groupby('exp_hh_match')['hh_vac_total_n'].sum() / df_companies['hh_vac_total_n'].sum())
print(df_companies.groupby('exp_n_staff_cleaned_interval')['hh_vac_total_n'].median())
print(df_companies.groupby('exp_n_staff_cleaned_interval')['hh_vac_total_n'].mean())

# Remove companies without a match
df_companies = df_companies[df_companies['exp_hh_match']]
print(df_companies['exp_revenue_2020'].notna().sum() / len(df_companies))
print(df_companies['exp_okved_code'].notna().sum() / len(df_companies))
print(df_companies['exp_n_staff_cleaned_numeric'].notna().sum() / len(df_companies))

# Compare OKVED between exportbase and matched hh
table_okved_comparision = pd.DataFrame(
    df_companies_exportbase_short['exp_okved_name_2'].value_counts(normalize=True)).reset_index()
table_okved_comparision = table_okved_comparision.merge(pd.DataFrame(
    df_companies['exp_okved_name_2'].value_counts(normalize=True)).reset_index(), on='index', how='left')
table_okved_comparision.columns = ['okved_name_2', 'pct_exp', 'pct_hh']
table_okved_comparision['rate_hh_exp'] = table_okved_comparision['pct_hh'] / table_okved_comparision['pct_exp'] 
table_okved_comparision.corr()


# =============================================================================
#
# --- 5. MATCH WITH VACANCIES DATA
#
# =============================================================================

# Read from pickle
os.chdir("/data")
df_hh = pd.read_pickle("df_hh_2019_2020_skills_isco.obj")
df_companies = pd.read_pickle('df_companies.obj')
df_companies = df_companies[(df_companies['exp_okved_name_2'] != "Деятельность по трудоустройству и подбору персонала")] # < Remove HR agencies

# Merge companies with HH vacancies
df_companies_tomerge = df_companies[['hh_id', 'hh_region', 'exp_registration_year',
                                    'exp_revenue_2020', 'exp_revenue_2019', 'exp_revenue_2018', 'exp_revenue_2017',
                                    'exp_profit_2020', 'exp_profit_2019', 'exp_profit_2018', 'exp_profit_2017',
                                    'exp_n_staff_cleaned_numeric', 'exp_n_staff_cleaned_interval',
                                    'exp_region_cleaned', 'exp_okved_code_2', 'exp_okved_name_2', 'exp_hh_match', 'exp_coordinates'
                                    ]]
df_hh = df_hh.merge(df_companies_tomerge, how='left',
                    left_on='employer_id', right_on='hh_id')

print(df_hh['hh_region'].notna().sum() / len(df_hh))
employer_id_matched = set(df_hh[df_hh['hh_region'].notna()]['employer_id'].unique())
employer_id_total = set(df_hh['employer_id'].unique())
print(len(employer_id_matched) / len(employer_id_total))

# Remove vacancies without a match
df_hh = df_hh[df_hh['hh_region'].notna()]

### EDA

# 1. TOP-3 ISCO for each OKVED
cross_isco_okved = pd.crosstab(df_hh['isco_name_3'], df_hh['exp_okved_name_2'], normalize='index')
vec_top3_isco = []
for i in range(len(cross_isco_okved.columns)):
    vec_top3_isco.append(list(cross_isco_okved.iloc[:,i].nlargest(3).index))
df_top3_isco = pd.DataFrame(vec_top3_isco)
df_top3_isco.index = cross_isco_okved.columns
df_top3_isco.reset_index(inplace=True)

# 2. Compare OKVED between exportbase and matched hh
df_hh_employer_id = df_hh.drop_duplicates('employer_id')
table_okved_comparision = pd.DataFrame(
    df_companies_exportbase_short['exp_okved_name_2'].value_counts(normalize=True)).reset_index()
table_okved_comparision = table_okved_comparision.merge(pd.DataFrame(
    df_hh_employer_id['exp_okved_name_2'].value_counts(normalize=True)).reset_index(), on='index', how='left')
table_okved_comparision.columns = ['okved_name_2', 'pct_exp', 'pct_hh']
table_okved_comparision['rate_hh_exp'] = table_okved_comparision['pct_hh'] / table_okved_comparision['pct_exp'] 
exp_okved_name_2_counts = pd.DataFrame(
    df_companies_exportbase_short['exp_okved_name_2'].value_counts(normalize=False)).reset_index()
exp_okved_name_2_counts.columns = ['okved_name_2', 'count_hh']
table_okved_comparision = table_okved_comparision.merge(exp_okved_name_2_counts)

# 3. Salaries by OKVED
t = pd.DataFrame(df_hh.groupby('exp_okved_name_2')['salary_mean'].mean()).reset_index()

### COORDINATES
df_hh_address = pd.read_pickle("df_hh_address.obj")
df_hh = df_hh.merge(df_hh_address, how='left')


def extractHhCoordinates(json_address):
    
    result = np.nan
    if type(json_address) == dict:
        if "lat" in json_address:
            result = str(json_address['lat']) + " " + str(json_address['lng'])
            if result in set(["None None", "0.0 0.0"]):
                result = np.nan
    
    return result
    

df_hh['hh_coordinates'] = df_hh['address'].apply(extractHhCoordinates)


def mergeCoordinates(row):
    if type(row['hh_coordinates']) == str:
        result = row['hh_coordinates']
    elif type(row['exp_coordinates']) == set:
        result = ', '.join(list(row['exp_coordinates']))
    else:
        result = np.nan
        
    return result


df_hh['coordinates'] = df_hh[['exp_coordinates', 'hh_coordinates']].apply(mergeCoordinates, 1)

# Save to pickle
os.chdir("/data")
df_hh.to_pickle('df_hh_2019_2020_skills_isco_coords.obj')



