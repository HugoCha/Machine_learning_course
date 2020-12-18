#! /bin/env python3.6

import os, sys
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import re
import numpy as np
from collections import Counter
import nltk
from nltk.stem.porter import PorterStemmer

######## All useful regex pattern for heading preprocessing ########
received_pattern = re.compile("^Received:.*\n\t.*\n\t.*\n|^Received:.*\n\t.*\n|^Received:.*\n",
                              flags=re.M|re.X)
messageId_pattern = re.compile("^Message-ID:.*\n",
                              flags=re.M|re.X)
returnpath_pattern = re.compile("^Return-Path:.*\n",
                              flags=re.M|re.X)
from_pattern = re.compile("^From[:\s].*\n",
                         flags=re.M|re.X)
deliver_pattern = re.compile("^Delivered-To:.*\n", 
                            flags=re.M|re.X)
subject_pattern = re.compile("^Subject:.*\n", 
                            flags=re.M|re.X)
date_pattern = re.compile("^Date:.*\n", 
                            flags=re.M|re.X)
content_type_pattern = re.compile("^Content-Type:.*\n", 
                            flags=re.M|re.X)
content_transfer_pattern = re.compile("^Content-Transfer-Encoding:.*\n", 
                            flags=re.M|re.X)
content_pattern = re.compile("^Content-.*:.*\n",
                            flags=re.M|re.X)
mime_pattern = re.compile("^MIME-Version:.*\n", 
                            flags=re.M|re.X)
bcc_pattern = re.compile("^Bcc:.*\n", 
                            flags=re.M|re.X)
importance_pattern = re.compile("^Importance:.*\n", 
                            flags=re.M|re.X)
to_pattern = re.compile("^To:.*\n", 
                            flags=re.M|re.X)
xmailer_pattern = re.compile("^X-Mailer:.*\n", 
                            flags=re.M|re.X)
xpriority_pattern = re.compile("^X-Priority:.*\n", 
                            flags=re.M|re.X)
x_spam_pattern = re.compile("^X-Spam:.*\n",
                           flags=re.M|re.X)
x_pattern = re.compile("^X-.*:.*\n", 
                            flags=re.M|re.X)
replyto_pattern = re.compile("^Reply-To:.*\n", 
                            flags=re.M|re.X)
list_pattern = re.compile("^List-.*:.*\n", 
                            flags=re.M|re.X)
errors_pattern = re.compile("^Errors:.*\n", 
                            flags=re.M|re.X)
hyphen_to_pattern = re.compile("^[\w\-]+-To:.*\n", 
                            flags=re.M|re.X)

heading_pattern = [
    received_pattern, 
    messageId_pattern,
    returnpath_pattern,
    from_pattern,
    deliver_pattern,
    subject_pattern,
    date_pattern,
    content_type_pattern,
    content_transfer_pattern,
    mime_pattern,
    bcc_pattern,
    importance_pattern,
    to_pattern,
    xmailer_pattern,
    xpriority_pattern,
    x_spam_pattern,
    x_pattern,
    replyto_pattern,
    list_pattern,
    errors_pattern,
    hyphen_to_pattern,
    ]


###### All useful patterns for preprocessing content ######
HTML_pattern = re.compile("\<.*?\>", re.S)
URL_pattern = re.compile("http.*\s", re.I | re.X)
mail_pattern = re.compile("\w+@[\w\.]+\W", re.I)
arrow_pattern = re.compile("^\>", re.M)
price_pattern = re.compile("[\$£]\s{0,2}\d+[,\.]\d+ | [\$£]\s{0,2}\d+ | \d+\s{0,2}[$£] | \d+[,\.]\d+\s{0,2}[$£]", re.X)
one_two_letter_word_pattern = re.compile(r"\b[a-zA-Z]{1,2}\b", re.X | re.M)

punctuation_pattern = re.compile(r"[\?\!]", re.X)
hour_date_pattern = re.compile("\d{4}\s*[a-zA-Z]{2,8}\s*\d{1,2} | \d{1,2}\s*[a-zA-Z]{2,8}\s*\d{4} | \d{1,2}:\d{2}:\d{2}", re.X)

maj_pattern = re.compile("[A-Z]", re.X)
number_pattern = re.compile("\d+", re.X)

word_pattern = re.compile("\\b[\w\+\-\=\&\*]+\\b", re.I | re.X)

most_used_word_subject = [
'perl',
'free',
'user',
'from',
'and',
'for',
'get',
'sadev',
'new',
'use',
'razor',
'spam',
'you',
'spambay',
'ilug',
'best',
'satalk',
'with',
'adv',
'the',
'mortgag',
'apt',
'your',
'spamassassin',
'wa',
'onlin']

most_used_word_content = [
'phone',
'had',
'for',
'there',
'type',
'get',
'should',
'not',
'busi',
'new',
'them',
'think',
'web',
'about',
'world',
'govern',
'just',
'date',
'with',
'the',
'other',
'url',
'see',
'thi',
'receiv',
'size',
'wa',
'pleas',
'becaus',
'need',
'are',
'click',
'compani',
'linux',
'even',
'who',
'could',
'from',
'text',
'have',
'they',
'some',
'one',
'also',
'grant',
'would',
'home',
'peopl',
'use',
'email',
'you',
'ani',
'make',
'over',
'cfont',
'out',
'much',
'inform',
'their',
'your',
'spamassassin',
'into',
'list',
'more',
'order',
'nbsp',
'wrote',
'don',
'work',
'call',
'way',
'free',
'mail',
'and',
'can',
'then',
'ha',
'ffont',
'send',
'face',
'now',
'what',
'time',
'remov',
'here',
'been',
'our',
'internet',
'than',
'onli',
'form',
'how',
'it',
'will',
'user',
'which',
'want',
'know',
'were',
'but',
'most',
'that',
'name',
'content',
'money',
'well',
'all',
'address',
'may',
'messag',
'color',
'when',
'help',
'like']

def isolate_all_word(content):
    """
    Isolate content words
    """
    pattern = re.compile("\w+", re.I | re.X | re.M)
    new_content = pattern.findall(content)
    return new_content

def count_common(content, list2compare):
    """
    Count the common word with the predifined lists most common word
    """
    word_list = isolate_all_word(content)
    word_list_length = len(word_list)
    if len(word_list) != 0:
        counting_list = [round(word_list.count(x)/len(word_list), 4) for x in list2compare]
    else:
        counting_list = [0 for x in list2compare]
    return counting_list



######### A general class for processing heading #########
class HeadPattern():
    def __init__(self, pattern):
        self.pattern = pattern
        self.pattern_list = []
    
    @property
    def pattern_cat(self):
        return 0    
    
    def find(self, header):
        self.pattern_list = self.pattern.findall(header)
        
    def isNone(self):
        return (self.pattern_list == [])
    
    def display_pattern(self):
        print(self.pattern_list)
    
    def len_list_pattern(self):
        return (len(self.pattern_list))
    
    def processing(self, header):
        pass

######### A general class for processing content #########
class ContentPattern():
    def __init__(self, pattern):
        self.pattern = pattern
        self.pattern_list = []
    
    @property
    def pattern_cat(self):
        return 0    
    
    def find(self, content):
        self.pattern_list = self.pattern.findall(content)
        
    def isNone(self):
        return (self.pattern_list == [])
    
    def display_pattern(self):
        print(self.pattern_list)
    
    def len_list_pattern(self):
        return (len(self.pattern_list))
    
    def count(self, content):
        self.find(content)
        return (self.len_list_pattern())
    
    def replace(self, content):
        new_content = self.pattern.sub("", content)
        return (new_content)
    
    def count_replace(self, content):
        pattern_nb = 0
        new_content, pattern_nb = self.pattern.subn("", content)
        return (new_content, pattern_nb)
    
    def processing(self, content):
        pass


######### Herited From Head Pattern to process heading #########
class FromPattern(HeadPattern):
    def __init__(self):
        HeadPattern.__init__(self, from_pattern)
    
    def processing(self, header):
        self.find(header)
        name = ''
        domain = ''
        extension = ''
        if (self.isNone() == False):
            length = self.len_list_pattern()
            for i in range(length):
                mail_pattern1 = re.compile("(From:\s)<?(?P<name>.*)@(?P<domain>.*)\.(?P<extension>[\w\.]*)>?", re.I)
                #mail_pattern2 = re.compile("(From:\s)(?P<name>.*)@(?P<domain>.*)\.(?P<extension>.*)", re.I)
                mail_address = mail_pattern1.search(self.pattern_list[i])
                #if (mail_address == None):
                #    mail_address = mail_pattern2.search(self.pattern_list[i])
                if (mail_address != None):
                    name = mail_address.group('name')
                    domain = mail_address.group('domain')
                    extension = mail_address.group('extension')
        len_name = len(name)
        number_in_name = re.findall("\d", name)
        if (number_in_name != None):
            len_number_in_name = len(re.findall("\d", name))
        else:
            len_number_in_name = 0
        return ([len_name, len_number_in_name, domain, extension])
    
    @property
    def pattern_cat(self):
        return ("from")

class DatePattern(HeadPattern):
    def __init__(self):
        HeadPattern.__init__(self, date_pattern)
    
    def processing(self, header):
        self.find(header)
        hour = np.nan
        if (self.isNone() == False):
            hour_patt = re.compile("(?P<hour>\d{1,2}):(?P<minute>\d{2}):(?P<second>\d{2})")
            hour_search = hour_patt.search(self.pattern_list[0])
            hour = int(hour_search.group('hour')) if (hour_search!=None) else np.nan
        return (hour)
    @property
    def pattern_cat(self):
        return("hour")

class ContentTypePattern(HeadPattern):
    def __init__(self):
        HeadPattern.__init__(self, content_type_pattern)
    
    def processing(self, header):
        self.find(header)
        content_type = np.nan
        if (self.isNone() == False):
            content_type_patt = re.compile("""(Content-Type:)\s(?P<type>.*)[;\n]""", re.X)
            content_type_match = content_type_patt.search(self.pattern_list[0])
            content_type = content_type_match.group('type') if (content_type_match!=None) else np.nan
        return (content_type)
    
    @property
    def pattern_cat(self):
        return("content-type")

class ImportancePattern(HeadPattern):
    def __init__(self):
        HeadPattern.__init__(self, importance_pattern)
    
    def processing(self, header):
        self.find(header)
        importance = np.nan
        if (self.isNone() == False):
            importance = self.pattern_list[0][12:]
        return (importance)
    @property
    def pattern_cat(self):
        return("importance")

class XSpamPattern(HeadPattern):
    def __init__(self):
        HeadPattern.__init__(self, x_spam_pattern)
    
    def processing(self, header):
        self.find(header)
        x_spam_word = np.nan
        if (self.isNone() == False):
            x_spam_word = self.pattern_list[0][8:]
        return (x_spam_word)
    
    @property
    def pattern_cat(self):
        return("x-spam")

class X_Pattern(HeadPattern):
    def __init__(self):
        HeadPattern.__init__(self, x_pattern)
    
    def processing(self, header):
        self.find(header)
        return (self.len_list_pattern())
    
    @property
    def pattern_cat(self):
        return("x-*")


######### Class for processing subject #########
class SubjectPattern(HeadPattern):

    def __init__(self):
        HeadPattern.__init__(self, subject_pattern)
    
    def processing(self, header):
        self.find(header)
        number_of_maj = 0
        number_of_number = 0
        number_of_char = 0
        number_of_price = 0
        number_of_special_char = 0
        common_word_list = [0 for x in most_used_word_subject]
        if (self.isNone() == False):
            # Extract info on number of char in function of categories (maj, punctuation)
            number_of_maj = len(re.findall("[A-Z]", self.pattern_list[0])) - 1 #Subtract "S" of "Subject"
            number_of_char = len(self.pattern_list[0]) - 8 #Subtract len("Subject:"))
            number_of_special_char = len(re.findall("[\!\?\$]", self.pattern_list[0]))
            
            # Extract info on number of char in function of categories (price,number)
            # And replace by "" those pattern
            new_subject, number_of_price = PricePattern().processing(self.pattern_list[0])
            new_subject, number_of_number = NumberPattern().processing(new_subject)
            # Replace all one or two letters word by ""
            new_subject = OneTwoLetterWordPattern().processing(new_subject)
            
            # Extact all remaining words except Subject of course
            new_subject = new_subject[8:]
            common_word_list = count_common(new_subject, most_used_word_subject)
        
        return ([number_of_maj, 
                 number_of_char, 
                 number_of_special_char, 
                 number_of_price, 
                 number_of_number, 
                 common_word_list])
    
    @property
    def pattern_cat(self):
        return("subject")


######### Herited From Content Pattern to process content #########
class HTMLPattern(ContentPattern):
    def __init__(self):
        ContentPattern.__init__(self, HTML_pattern)
    
    def processing(self, content):
        return self.count_replace(content)
    @property
    def pattern_cat(self):
        return("html")

class URLPattern(ContentPattern):
    def __init__(self):
        ContentPattern.__init__(self, URL_pattern)
    
    def processing(self, content):
        return self.count_replace(content)
    
    @property
    def pattern_cat(self):
        return("url")

class MailPattern(ContentPattern):

    def __init__(self):
        ContentPattern.__init__(self, mail_pattern)
    
    def processing(self, content):
        return self.count_replace(content)
    
    @property
    def pattern_cat(self):
        return("mail")

class ArrowPattern(ContentPattern):
    def __init__(self):
        ContentPattern.__init__(self, arrow_pattern)
    
    def processing(self, content):
        return self.count_replace(content)
    
    @property
    def pattern_cat(self):
        return("> count")

class PricePattern(ContentPattern):
    def __init__(self):
        ContentPattern.__init__(self, price_pattern)
    
    def processing(self, content):
        return self.count_replace(content)
    
    @property
    def pattern_cat(self):
        return("price")

class PunctuationPattern(ContentPattern):
    def __init__(self):
        ContentPattern.__init__(self, punctuation_pattern)
    
    def processing(self, content):
        return self.count_replace(content)
    
    @property
    def pattern_cat(self):
        return("? or !")

class NumberPattern(ContentPattern):
    def __init__(self):
        ContentPattern.__init__(self, number_pattern)
    
    def processing(self, content):
        return self.count_replace(content)
    
    @property
    def pattern_cat(self):
        return("Number count")

class HourDatePattern(ContentPattern):
    def __init__(self):
        ContentPattern.__init__(self, hour_date_pattern)
    
    def processing(self, content):
        return self.replace(content)

class OneTwoLetterWordPattern(ContentPattern):
    def __init__(self):
        ContentPattern.__init__(self, one_two_letter_word_pattern)
    
    def processing(self, content):
        return self.replace(content)

class MajPattern(ContentPattern):
    def __init__(self):
        ContentPattern.__init__(self, maj_pattern)
    
    def processing(self, content):
        return self.count(content)
    
    @property
    def pattern_cat(self):
        return("Majuscule count")

class WordPattern(ContentPattern):

    def __init__(self):
        ContentPattern.__init__(self, word_pattern)
    
    def processing(self, content):
        self.find(content)
        number_of_word = self.len_list_pattern()
        len_word_list = list(map(len, self.pattern_list))
        if len(len_word_list) != 0:
            mean_len_word = round(sum(len_word_list)/len(len_word_list), 4)
            longuest_word = max(len_word_list)
        else:
            mean_len_word = 0
            longuest_word = 0
        
        count_common_word = count_common(content, most_used_word_content)
        return ([number_of_word, mean_len_word, longuest_word, count_common_word])
    @property
    def pattern_cat(self):
        return("content_attr")
