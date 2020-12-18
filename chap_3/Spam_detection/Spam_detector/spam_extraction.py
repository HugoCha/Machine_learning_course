import pandas as pd
import re
import numpy as np
from ast import literal_eval
import scipy.stats as stats

from sklearn.base import TransformerMixin
from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn_utils import *

####### For Extraction of CSV #######
def string2list(train_set, cat):
    """
    Convert a string containing a list to a list 
    CSV import object as string
    """
    new_train_set = train_set.copy(deep=True)
    new_train_set[cat] = new_train_set[cat].apply(lambda x:literal_eval(x))
    return new_train_set

string2list_pipeline = Pipeline([
    ('str2l', FunctionTransformer(
        func=string2list, 
        kw_args={'cat':"from"})),
    ('str2l1', FunctionTransformer(
        func=string2list, 
        kw_args={'cat':"subject"})),
    ('str2l2', FunctionTransformer(
        func=string2list, 
        kw_args={'cat':"content_attr"})),
    ])

####### For splitting list into several features #######
def split_list(train_set, cat, names):
    """
    Split list to separate features
    """
    new_train_set = train_set.copy(deep=True)
    if (len(names) == len(new_train_set[cat].iloc[0])):
        i=0
        for name in names:
            new_train_set[name] = new_train_set[cat].apply(lambda x:x[i])
            i+=1
    return new_train_set.drop(cat, axis=1)

split_list_pipeline = Pipeline([
    ('splitl', FunctionTransformer(
        func=split_list, 
        kw_args={
            'cat':"from",
            'names' : ("len_mail_name", 
                       "number_mail_name", 
                       "domain",
                       "extension"),
            })),
    ('splitl1', FunctionTransformer(
        func=split_list, 
        kw_args={
            'cat':"subject",
            'names' : ( "subj_number_of_maj",
                        "subj_number_of_char",
                        "subj_number_of_special_char",
                        "subj_number_of_price",
                        "subj_number_of_number",
                        "subj_common_word_list"),
        })),
    ('splitl2', FunctionTransformer(
        func=split_list, 
        kw_args={
            'cat':"content_attr",
            'names' : ( "cont_number_of_word",
                        "cont_mean_len_word",
                        "cont_longuest_word",
                        "cont_count_common_word"),
        })),])

split_csv_list_pipeline = Pipeline([
    ('str2l', FunctionTransformer(
        func=string2list, 
        kw_args={'cat':"from"})),
    ('str2l1', FunctionTransformer(
        func=string2list, 
        kw_args={'cat':"subject"})),
    ('str2l2', FunctionTransformer(
        func=string2list, 
        kw_args={'cat':"content_attr"})),
    
    ('splitl', FunctionTransformer(
        func=split_list, 
        kw_args={
            'cat':"from",
            'names' : ("len_mail_name", 
                       "number_mail_name", 
                       "domain",
                       "extension"),
            })),
    ('splitl1', FunctionTransformer(
        func=split_list, 
        kw_args={
            'cat':"subject",
            'names' : ( "subj_number_of_maj",
                        "subj_number_of_char",
                        "subj_number_of_special_char",
                        "subj_number_of_price",
                        "subj_number_of_number",
                        "subj_common_word_list"),
        })),
    ('splitl2', FunctionTransformer(
        func=split_list, 
        kw_args={
            'cat':"content_attr",
            'names' : ( "cont_number_of_word",
                        "cont_mean_len_word",
                        "cont_longuest_word",
                        "cont_count_common_word"),
        })),])


####### For Creating new, more relevant categories  #######
def divide_multiply(train_set, cat1, cat2, name, div=True):
    new_train_set = train_set.copy()
    if (div):
        new_train_set_cat = new_train_set[:, cat1]/new_train_set[:, cat2]
        new_train_set_cat[ ~ np.isfinite( new_train_set_cat )]  = 0
        #new_train_set = new_train_set.drop(cat1, axis=1)
        return np.c_[new_train_set[:, 0:cat1], new_train_set[:, cat1+1:], new_train_set_cat]
    else:
        new_train_set_cat = new_train_set[:, cat1]*new_train_set[:, cat2]
    return np.c_[new_train_set, new_train_set_cat]

def working_hour(train_set, index_hour):
    new_train_set = train_set.copy()
    new_train_set_working_hourless17 = new_train_set[:, index_hour] < 17
    new_train_set_working_hourup9 = new_train_set[:, index_hour] > 9
    new_train_set_working_hour = new_train_set_working_hourless17 & new_train_set_working_hourup9
    new_train_set_working_hour = new_train_set_working_hour.astype(int)
    return np.c_[new_train_set[:, :index_hour], new_train_set[:, index_hour+1:],new_train_set_working_hour]

index_hour = 0
index_html = 1
index_Number_count= 2
index_Majuscule_count = 3
index_subj_number_of_maj=4
index_cont_number_of_word=5
index_subj_number_of_special_char= 6

num_attribs = [
    'hour','html','Number count','Majuscule count','subj_number_of_maj',
    "cont_number_of_word",'subj_number_of_special_char',
    'x-*', '> count', '? or !', 'number_mail_name','subj_number_of_char',
    'subj_number_of_price','subj_number_of_number',
    'cont_mean_len_word', "cont_longuest_word",
]

num_pipeline = Pipeline([

    ('imputer', SimpleImputer(strategy='median')),
    
    ('divide', FunctionTransformer(
        func=divide_multiply, 
        kw_args={
            'cat1': index_Majuscule_count,
            'cat2' : index_cont_number_of_word,
            'name': "maj_over_count",
        })),
    ('divide1', FunctionTransformer(
        func=divide_multiply, 
        kw_args={
            'cat1':index_Number_count,
            'cat2' : index_cont_number_of_word,
            'name': "num_over_count",
        })),
    ('divide2', FunctionTransformer(
        func=divide_multiply, 
        kw_args={
            'cat1': index_html,
            'cat2':  index_cont_number_of_word,
            'name': "html_over_count",
        })),
    ('multiply', FunctionTransformer(
        func=divide_multiply, 
        kw_args={
            'cat1': index_subj_number_of_maj,
            'cat2' : index_subj_number_of_special_char,
            'name': "subj_maj_time_spechar",
            'div' : False,
        })),
    
    ('working_hour', FunctionTransformer(
        func=working_hour,
        kw_args={
            'index_hour':index_hour,
        }
    )),
    ("scaler", StandardScaler()),])

####### For separating domains into bins #######
l_1020 = ['lycos', 'ximian', 'eecs.berkeley', 'maxtor', 'terra', 'waider', 'linuxworks.com',
 'caramail', 'ygingras', 'earthlink', 'cs.Helsinki', 'ckloiber', 'urgent.rug.ac', 'bennewitz',
 'mediaunspun.imakenews', 'wstoddard', 'lin12.triumf', 'mindspring', 'woozle', 'cs.helsinki',
 'freemail', 'punkass', 'noskillz', 'wanadoo', 'med.wayne', 'interlink.com', 'mad.scientist',
 'perkel', 'zanshin', 'FrugalJoe', 'email', 'netnoteinc', 'freeuk', 'kbs', 'paradigm-omega',
 'greenhydrant', 'deersoft', 'neo.pittstate', 'gmx', 'frogstone', 'ummail4.unitedmedia',
 'bluemail', 'baesystems', 'svanstrom', 'ie.suberic', 'list.theregister.co', 'Golux', 'alumni.rice',
 'ianbell', 'cunniffe', 'dmv', 'whump', 'lig', 'iol', 'bubbanfriends']
l_2030 = ['dogma.slashnull', 'netscape', 'talios', 'corvil', 'sendgreatoffers', 'srv0.ems.ed.ac',
 'eudoramail', 'kamakiriad', 'Flashmail', 'panix', 'excite', 'users.sourceforge', 'bellsouth',
 'redbrick.dcu', 'fastmail', 'pathname', 'kluge', 'vipul', 'linuxmafia', 'btamail.net']
l_sup30 = ['egwn','example', 'magnesium', 'aol', 'yahoo.co', 'endeavors', 'comcast', 'insurancemail',
 'acm', 'perl', 'iki', 'munnari.OZ', 'python', 'hotmail', 'spamassassin.taint', 'newsletter.online',
 'evergo', 'techmonkeys', 'alumni.caltech', 'barrera', 'panasas', 'yahoo', 'lockergnome',
 'rpmforge', 'eircom', 'petting-zoo', 'mithral', 'aminvestments', 'pobox', '2ubh', 'leitl',
 'slack', 'shipwright', 'DeepEddy', 'cse.ucsc', 'canada', 'msn', 'hughes-family', 'insiq',
 'silcom', 'permafrost', 'tuatha', 'argote', 'usa', 'best', 'mail', 'linux', '10-20', '20-30']

def bins_domain(train_set):
    new_train_set = train_set.copy(deep=True)
    
    new_train_set["domain"] = np.where(np.invert(new_train_set["domain"].isin(l_1020)), new_train_set["domain"], '10-20')
    new_train_set["domain"] = np.where(np.invert(new_train_set["domain"].isin(l_2030)), new_train_set["domain"], '20-30')
    new_train_set["domain"] = np.where(new_train_set["domain"].isin(l_sup30), new_train_set["domain"], '0-10')
    #print(new_train_set["domain"].value_counts())
    return new_train_set

domain_attribs = ["domain"]
domains_cat =[l_sup30+["0-10"]]

bin_domain_pipeline = Pipeline([
    ('bins_domain', FunctionTransformer(func=bins_domain)),
    #('bins_extension', FunctionTransformer(func=bins_extension)),
    #('bins_content_type', FunctionTransformer(func=bins_content_type)), 
    ("one_hot",OneHotEncoder(handle_unknown="ignore", sparse=True, categories=domains_cat))])

####### For separating extension into bins #######
l_34 = ['pt', 'th', 'no', 'ph', 'sk', 'bz', 'tr']
l_56 = ['kr', 'co', 'ro', 'cc', 'gr', 'hu', 'pl']
l_78 = ['to']
l_910 = ['tw', 'nl', 'se']
l_sup_9 = ['ie', 'it', 'ru', 'at', 'es', 'org', 'us',
 'dk', 'nu', 'za', 'fm', 'cn', 'fi', 'fr', 'com', 'be',
 'de', 'net', 'ch', 'br', 'jp', 'au', 'uk', 'ca', 'edu']

def bins_extension(train_set):
    new_train_set = train_set.copy(deep=True)
    new_train_set["extension"] = np.where(np.invert(new_train_set["extension"].isin(l_34)), new_train_set["extension"], '3-4')
    new_train_set["extension"] = np.where(np.invert(new_train_set["extension"].isin(l_56)), new_train_set["extension"], '5-6')
    new_train_set["extension"] = np.where(np.invert(new_train_set["extension"].isin(l_78)), new_train_set["extension"], '7-8')
    new_train_set["extension"] = np.where(np.invert(new_train_set["extension"].isin(l_910)), new_train_set["extension"], '9-10')
    new_train_set["extension"] = np.where(new_train_set["extension"].isin(l_sup_9), new_train_set["extension"], '1-2')
    new_train_set["extension"] = new_train_set["extension"].str.lower()
    return new_train_set

extension_attribs = ["extension"]
extension_cat =[l_sup_9+["1-2"]]
bin_extension_pipeline = Pipeline([
    #('bins_domain', FunctionTransformer(func=bins_domain)),
    ('bins_extension', FunctionTransformer(func=bins_extension)),
    #('bins_content_type', FunctionTransformer(func=bins_content_type)), 
    ("one_hot",OneHotEncoder(handle_unknown="ignore", sparse=True, categories=extension_cat))])

####### For separating content-type into bins #######
def bins_content_type(train_set):

    new_content_type = train_set.copy(deep=True)
    new_content_type["content-type"] = new_content_type["content-type"].str.lower()
    new_content_type["content-type"] = new_content_type["content-type"].str.replace(";", " ", regex=True)
    new_content_type["content-type"] = new_content_type["content-type"].str.replace('\"', " ", regex=True)
    new_content_type["content-type"] = new_content_type["content-type"].fillna("unkown")
    new_content_type["content-type"] = new_content_type["content-type"].str.split()
    new_content_type["content-type"] = new_content_type["content-type"].apply(lambda x: x[0])
    #new_content_type["content-type"] = new_content_type["content-type"].str.lower()
    return new_content_type

content_attribs = ["content-type"]#, "extension", "content-type"
content_cat = [['text/plain','unkown','text/html','multipart/alternative',
               'multipart/signed','multipart/mixed','multipart/related','multipart/report']]  
bin_content_pipeline = Pipeline([
    #('bins_domain', FunctionTransformer(func=bins_domain)),
    ('bins_content_type', FunctionTransformer(func=bins_content_type)),
    #('bins_content_type', FunctionTransformer(func=bins_content_type)), 
    ("one_hot",OneHotEncoder(handle_unknown="ignore", sparse=True, categories=content_cat))])


####### For splitting list of words #######
def split_word_list(train_set, cat, name):
    new_train_set = train_set.copy(deep=True)
    if type(new_train_set[cat].iloc[0])==type([]):
        for i in range(len(new_train_set[cat].iloc[0])):
            new_train_set[name+str(i)] = new_train_set[cat].apply(lambda x:x[i])
        return new_train_set.drop(cat, axis=1)
    return new_train_set

word_list_attrib = ["subj_common_word_list", "cont_count_common_word"]

word_list_pipeline = Pipeline([
    ("content_list", FunctionTransformer(
        func=split_word_list,
        kw_args={
            'cat' : "cont_count_common_word",
            'name' : "cw"
        }
    )),
    ("subject_list", FunctionTransformer(
        func=split_word_list,
        kw_args={
            'cat' : "subj_common_word_list",
            'name' : "sw"
        }
    ))])


###### Final pipeline ######
full_pipeline = ColumnTransformer([
    ("Imputer", DataFrameImputer, []),
    ("num_pipeline", num_pipeline, num_attribs),
    ("domain_pipeline", bin_domain_pipeline, domain_attribs),
    ("extension_pipeline", bin_extension_pipeline, extension_attribs),
    ("content_pipeline", bin_content_pipeline, content_attribs),])







