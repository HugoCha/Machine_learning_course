import re
import pandas as pd
import numpy as np
import sys, os
import joblib

from spam_pattern import *
from spam_extraction import *


important_heading_pattern = [
    FromPattern(),
    SubjectPattern(),
    DatePattern(),
    ContentTypePattern(),
    X_Pattern(),
]

count_replace_content_pattern_list = [
    HTMLPattern(),
    URLPattern(),
    MailPattern(),
    PricePattern(),
    ArrowPattern(),
    PunctuationPattern(),
    NumberPattern(),    
]

replace_content_list = [
    OneTwoLetterWordPattern(),
    HourDatePattern(),
]

count_content_list = [
    MajPattern(),
    WordPattern(),
]

class MailPreprocessing():
    def __init__(self, filepath, labelised=False, spam=True):
        self.filepath = filepath
        self.labelised = labelised
        self.spam = spam
        
        self.extract_txt()
        self.good_separation = self.separate_header_content()

        self.mail_dataframe = self.create_dataframe_cat()
        self.mail_dataframe_cat = list(self.mail_dataframe.columns)

    def show_dataframe(self):
        print(self.mail_dataframe.head(1))

    def extract_txt(self):
        """
        Extract a raw text from a filepath
        """
        try:
            f = open(self.filepath, "r", encoding='utf8', errors='replace')#, 
            text = f.read()
            self.raw_text = text
        except:
            return False
        return True

    def separate_header_content(self):
        """
        Split header from content supposing there is a newline between header and content
        """
        separator_pattern = re.compile(r"\n\n")
        mail = separator_pattern.split(self.raw_text, maxsplit=1)
        if len(mail)==1:
            heading = mail[0]
            content=""
            print("No content detected in : ", self.filepath)
            return False
        else:
            heading = mail[0]
            content = mail[1]
        header_in_content = False
        while(content != "" and header_in_content):
            header_in_content = False
            for pattern in heading_pattern:
                if pattern.match(content) != None:
                    mail = separator_pattern.split(content, maxsplit=1)
                    header_in_content = True
                    content = mail[1]
                    heading = heading + mail[0]
        self.heading = heading
        self.content = content
        #print(self.content)
        return True

    def process_heading(self):
        """
        Process heading with all header dedicated class
        """
        attributes = []
        for important_pattern in important_heading_pattern:
            attributes.append(important_pattern.processing(self.heading))
        return attributes 

    def process_content(self):
        """
        Process content with all header dedicated class
        """
        attributes = []
        new_content = self.content
        for count_repl_patt in count_replace_content_pattern_list:
            new_content, count = count_repl_patt.processing(new_content)
            attributes.append(count)
        for repl_patt in replace_content_list:
            new_content = repl_patt.processing(new_content)
        for count_patt in count_content_list:
            count = count_patt.processing(new_content)
            attributes.append(count)
        return attributes


    def create_dataframe_cat(self):
        # Create Dataframe
        mail_dataframe = pd.DataFrame()
        
        # Add a spam or not cat
        if (self.labelised):
            mail_dataframe['spam']=[]
        
        # Add Heading Attributes
        for important_pattern in important_heading_pattern:
            mail_dataframe[important_pattern.pattern_cat]=[]
        
        # Add Content Attributes
        for count_repl_patt in count_replace_content_pattern_list:
            mail_dataframe[count_repl_patt.pattern_cat]=[]
        for count_patt in count_content_list:
            mail_dataframe[count_patt.pattern_cat]=[]
        
        return mail_dataframe


    def process_mail(self):
        """
        Process the mail and return a dataframe
        """
        if (self.good_separation):
            heading_attributes = self.process_heading()
            content_attributes = self.process_content()

            if (self.labelised):
                self.mail_attributes =  [int(self.spam)] + heading_attributes + content_attributes
            else:
                self.mail_attributes =  heading_attributes + content_attributes
            dict2add = dict(zip(self.mail_dataframe_cat, self.mail_attributes))
            self.mail_dataframe = self.mail_dataframe.append(dict2add, ignore_index=True)

            self.mail_dataframe = split_list_pipeline.fit_transform(self.mail_dataframe)
            self.mail_dataframe = word_list_pipeline.fit_transform(self.mail_dataframe)
        
            self.X = full_pipeline.fit_transform(self.mail_dataframe)
            if (self.labelised):
                self.y = int(self.spam)

            print("Mail was correctly extracted")
            return True
        return False
    
    def process_mail_from_csv(self):
        self.mail_dataframe = split_csv_list_pipeline.fit_transform(self.mail_dataframe)
        self.mail_dataframe = word_list_pipeline.fit_transform(self.mail_dataframe)
        
        self.X = Full_pipeline.fit_transform(self.mail_dataframe)
        self.y = int(self.spam)
    
    def append2dataframe(self, dataframe):
        return pd.concat(dataframe, self.mail_dataframe, axis=0)

    def predict(self, model):
        return model.predict(self.X)
    
    def predict_proba(self, model):
        return model.predict_proba(self.X)



if __name__ == "__main__":
    process=True
    filename = "extra_tree_model.sav"
    loade_model = joblib.load(filename)    
    if (len(sys.argv)==1):
        print("You must pass a filepath as argument")
        process=False
    elif (len(sys.argv)==2):
        new_mail = MailPreprocessing(sys.argv[1])
    elif (len(sys.argv)==3):
        if not(bool(sys.argv[2])):
            new_mail = MailPreprocessing(sys.argv[1])
        else:
            print("You must enter the label as argument")
    else:
        new_mail = MailPreprocessing(sys.argv[1], bool(sys.argv[2]), bool(sys.argv[3]))
    
    if (process):
        new_mail.process_mail()
        y_pred = new_mail.predict_proba(loade_model)
        print(y_pred)
        #new_mail.show_dataframe()
        