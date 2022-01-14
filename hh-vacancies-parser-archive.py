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

os.chdir('/data')


# Parse vacancy
def getVacancy(vac_id):
    '''

    Parameters
    ----------
    vac_id : str
        ID of a vacancy.

    Returns
    -------
    dict
        JSON-type format object with information about a vacancy.

    '''
    
    url_vac = 'https://api.hh.ru/vacancies/%s'
    try:
        r = requests.get(url_vac % vac_id, timeout=30)
        result = json.loads(r.text)
    except:
        time.sleep(5)
        result = np.nan
    return result


def main(vac_ids):
    
    pool = multiprocessing.Pool(12)
    results = pool.map(getVacancy, vac_ids)

    pool.close()
    pool.join()
        
    return results


if __name__ == '__main__':
    
    vacancies_ids = list(map(str, np.arange(0, 42200000)))
    vacancies_ids_start = vacancies_ids[::100000][:-1]
    vacancies_ids_start = list(map(int, vacancies_ids_start))
    vacancies_ids_end = vacancies_ids[::100000][1:]
    vacancies_ids_end = list(map(int, vacancies_ids_end))

    for i in range(len(vacancies_ids_start)):
        vacancies_batch = main(list(map(str, np.arange(vacancies_ids_start[i],
                                                       vacancies_ids_end[i]))))
        file_batch = open('df_archive_{}_{}'.format(str(vacancies_ids_start[i]),
                                                   str(vacancies_ids_end[i])), 'wb')
        pickle.dump(vacancies_batch, file_batch)
        print(i)
        time.sleep(60)

















