# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 10:08:01 2018

@author: mzhen
"""

import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
# df = pd.read_excel(open('D:\Python Projects\Data Science\Data\Data Analysis Exercise - Source Data (July 2015).xlsx'),sheetname = 'Source Data')

Df_practice = pd.read_excel('D:\Python Projects\Data Science\Data\Practice Data.xlsx', sheetname = 'Sheet1')
Df_data = pd.read_excel('D:\Python Projects\Data Science\Data\Data Analysis Exercise - Source Data (July 2015).xlsx',sheetname = 'Source Data')
