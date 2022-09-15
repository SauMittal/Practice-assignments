# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 19:13:16 2022

@author: kiit
"""

import json 
import pandas as pd 
import numpy as np 
 

test = pd.read_json("C:/Users/kiit/Downloads/test.json")
test.sort_values('ingredients')['ingredients'].drop_duplicates().reset_index(drop = True)

a_array= np.array(test['ingredients'])
ingredient_dictionary = {}

for a_list in a_array:
    for a in a_list:
        if a in ingredient_dictionary.keys():
            ingredient_dictionary[a] += 1 
        else:
            ingredient_dictionary[a] = 1
    
ingredient_dictionary
num_ing = len(ingredient_dictionary.keys())
print(num_ing)

total_count = 0 
for count in ingredient_dictionary.values():
    total_count += count
avg_count = total_count/num_ing
print(avg_count)
print(total_count)

count_array = np.array(list(ingredient_dictionary.values()))
np.where(count_array == 1)