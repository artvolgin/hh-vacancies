# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 19:50:26 2021

@author: Artem
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 02:29:28 2019

@author: Artem
"""

import requests
import pandas as pd
import numpy as np
import os
import pickle
from datetime import datetime
import time
import json
import gc
import schedule
import multiprocessing

os.chdir("/data")



def getCompany(company_id):
    '''

    Parameters
    ----------
    company_id : str
        ID of a company.

    Returns
    -------
    dict
        JSON-type format object with information about a company.

    '''
    
    url_vac = 'https://api.hh.ru/employers/%s'
    
    try:
        r = requests.get(url_vac % company_id)
        result = json.loads(r.text)
    except:
        time.sleep(2)
        result = np.nan
    return result



def main(companies_ids):
    
    pool = multiprocessing.Pool(12)
    results = pool.map(getCompany, companies_ids)

    pool.close()
    pool.join()
        
    return results


if __name__ == '__main__':
    
    companies_ids = list(pd.read_pickle(r'companies_ids.pickle'))   
    companies_ids_start = np.arange(len(companies_ids))[::10000][:-1]
    companies_ids_end = np.arange(len(companies_ids))[::10000][1:]
    
    for i in range(len(companies_ids_start)):
        companies_batch = main(companies_ids[companies_ids_start[i]:companies_ids_end[i]])
        file_batch = open('companies_info_{}_{}'.format(str(companies_ids_start[i]),
                                                   str(companies_ids_end[i])), 'wb')
        pickle.dump(companies_batch, file_batch)
        print(i)
        time.sleep(10)








