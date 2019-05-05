#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  5 13:15:25 2019

@author: lg
"""

#import jieba

docu_set={'d1':'i love shanghai',
          'd2':'i am from shanghai now i study in tongji university',
          'd3':'i am from lanzhou now i study in lanzhou university of science  and  technolgy',}


all_words=[]
for i in docu_set.values():
#    cut = jieba.cut(i)
    cut=i.split()
    all_words.extend(cut)
    
set_all_words=set(all_words)

print(set_all_words)


invert_index=dict()
for b in set_all_words:
    
    
    temp=[]
    for j in docu_set.keys():
        
        field=docu_set[j]
        
        split_field=field.split()
        
        if b in split_field:
            temp.append(j)
    invert_index[b]=temp
    
            
print(invert_index)            
        
    