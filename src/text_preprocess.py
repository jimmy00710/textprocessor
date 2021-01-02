import pandas as pd
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk import tokenize
from nltk.stem import WordNetLemmatizer 
from nltk.probability import FreqDist
from nltk.stem import WordNetLemmatizer 
from nltk.stem import PorterStemmer
import re
import nltk

from wordspellchecker import spell_checker_word
from contractmapping import contraction_mapping

stop_words = stopwords.words('english')

class NLP_Preprocessor:
    def __init__(self,
                 df):
    
        self.df = df
        self.fdist = FreqDist()
    
    
    def pipeline(self,processing_col):

        '''
        This is the default pipeline we can use in most of the text data. 
        In this first we are lowercasing the text data, after lower case we are removing the url, html tag and punctuations.
        Once punctuations are removed, we are removing the emojis. After removal of emojis we are tokenizing it into word tokens, And lemmatizing it. 
        Once lemma is created we are removing the stop words and after that we are calculating the frequency distribution. 
        Based on frequency distribution we are making 2 columns - rm_freq_word and rm_rare_word in both of them we are trying to remove frequent word and rare words.
        
        We can further add contract expansion and spell checking based on the project we are doing. 
        '''
        self.lowercase(processing_col,'lowercase')
        self.url_removal('lowercase','urlremoval')
        self.html_tag('urlremoval','html_tag')
        self.rm_punc('html_tag','rmpunc')
        self.remove_emoji('rmpunc','rmemoji')
        self.word_token('rmemoji','tokens')
        self.lemmatiz('tokens','lemma')
        self.rm_stopwords('lemma','rmstopwords')
        self.freq_words('rmstopwords')
        self.rm_freq_rare_words_('rmstopwords')
    
    def lowercase(self,
                   processing_col,
                   new_col):

        #Input - Take string 
        #Output - String
        self.df[new_col] = self.df[processing_col].str.lower()
    
    def url_removal(self,
                     processing_col,
                     new_col):

        #Input - String 
        #Output - String
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        self.df[new_col] = self.df[processing_col].apply(lambda text: url_pattern.sub(r'',text))
        
    def html_tag(self,
                  processing_col,
                  new_col):

       #Input - String 
       #Output - String 
        html_pattern = re.compile('<.*?>')
        self.df[new_col] = self.df[processing_col].apply(lambda text: html_pattern.sub(r'',text))
    
    
    def rm_punc(self,
                processing_col,
                new_col,
                punct_remove='!"#$%&\'()*+,-./:;<=>?@[\\]^_{|}~\n`'):
        
        #Input - String 
        #Output - String 
        self.df[new_col] = self.df[processing_col].apply(lambda text: text.translate(str.maketrans('','',punct_remove)))
    
    def rm_stopwords(self,
                      processing_col,
                      new_col,
                      stopword_list=[],):

        #Input - String 
        #Output - List of tokens 

        #The function accept the text data, flag to remove default stop word which are available in nltk.
        #And at last it accept stopword_list which can be empty
        stop_words = set(stopwords.words('english'))
        #if stopword_list and add=True:
            #Adding both the stop words
        
        self.df[new_col] = self.df[processing_col].apply(lambda text: ' '.join(word for word in text.split() if word not in stop_words))
        self.df[new_col] = self.df[processing_col].apply(lambda text: text.split(' '))
    
    def word_token(self,
                    processing_col,
                    new_col):
        #Function to convert all the data into tokens.

        #Input - String 
        #Output - List of tokens 
        self.df[new_col] = self.df[processing_col].apply(lambda text: nltk.tokenize.word_tokenize(text))
    
    def sent_tokens(self,
                     processing_col,
                     new_col):
        #Function to convert all the data into sentence tokens
        self.df[new_col] = self.df[processing_col].apply(lambda text: nltk.tokenize.sent_tokenize(text))
    
    
    def lemmatiz(self,
               processing_col,
               new_col):

        #Input - List of tokens 
        #Output - String 

        #Function to apply lemmatization
        lemmatizer = WordNetLemmatizer() 
        self.df[new_col] = self.df[processing_col].apply(lambda tokens : " ".join(lemmatizer.lemmatize(token) for token in tokens ))
        
    def stemmer(self,
                 processing_col,
                 new_col):

        #Input - List of Tokens 
        #Output - String 
        #Function to apply stemmer
        ps = PorterStemmer()
        self.df[new_col] = self.df[processing_col].apply(lambda tokens : " ".join(ps.stem(token) for token in tokens))
                
    
    def spell_correct_(self,):
        #Input - List of Token 
        #Output - List of token

        #Function to check the spelling.
        self.df['spell_corrected'] = self.df[self.col_name].apply(lambda token_list : spell_checker(token_list))
       
    
    
        
    def expand_the_contraction(self):
        
        #Input - List of tokens 
        #Output - String 

        #Function to expand the contractions.
        self.df['expanded'] = self.df[self.col_name].apply(lambda tokens: " ".join(contraction_mapping(token) for token in tokens))
                                                           
    def freq_words(self,
                  processing_col):
        for token_list in list(self.df[processing_col]):
            for token in token_list:
                self.fdist[token] += 1
    
    def rm_freq_rare_words_(self,
                            preprocessing_col,
                            new_col_freq_words='rm_freq_word',
                            new_col_rare_words='rm_rare_words',
                            top_freq_percent=10,
                            top_rare_percent=1):
        
        #Function to remove frequent word or rare word depending upon the flag
        
        #Input - List of Tokens 
        #Output - List of Tokens 
        
        top_freq_words = []
        top_rare_words = []
        
        for token_list in list(self.df[preprocessing_col]):
            for token in token_list:
                if self.fdist[token] >= top_freq_percent:
                    top_freq_words.append(token)
                elif self.fdist[token] <= top_rare_percent:
                    top_rare_words.append(token)
        self.df[new_col_freq_words] = self.df[preprocessing_col].apply(lambda tokens : " ".join(token for token in tokens if token not in top_freq_words))
        self.df[new_col_freq_words] = self.df[new_col_freq_words].apply(lambda text: text.split(' '))
        
        self.df[new_col_rare_words] = self.df[preprocessing_col].apply(lambda tokens : " ".join(token for token in tokens if token not in top_rare_words))
        self.df[new_col_rare_words] = self.df[new_col_rare_words].apply(lambda text: text.split(' '))    
    
    def remove_emoji(self,
                     processing_col,
                     new_col):
        #https://gist.github.com/slowkow/7a7f61f495e3dbb7e3d767f97bd7304b

        #Input - String 
        #Output - String 

        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"  # emoticons
                                   u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                   u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                   u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                   u"\U00002500-\U00002BEF"  # chinese char
                                   u"\U00002702-\U000027B0"
                                   u"\U00002702-\U000027B0"
                                   u"\U000024C2-\U0001F251"
                                   u"\U0001f926-\U0001f937"
                                   u"\U00010000-\U0010ffff"
                                   u"\u2640-\u2642"
                                   u"\u2600-\u2B55"
                                   u"\u200d"
                                   u"\u23cf"
                                   u"\u23e9"
                                   u"\u231a"
                                   u"\ufe0f"  # dingbats
                                   u"\u3030"
                                   u"\u2022"
                                   "]+", flags=re.UNICODE)
        #return emoji_pattern.sub(r'', string)
        self.df[new_col] = self.df[processing_col].apply(lambda text: emoji_pattern.sub(r' ',text))
        
    def conv_emoji(self,cnt_list,rm_emoji_fg,rm_emoticon_fg,cnv_emoji_fg,cnv_emoticon_fg):
        #Function to either remove emoji and emoticion or convert them into words.
        pass
    
    def chat_conv(self,
                df,
                col_name):
        #Funciton convert the chat
        pass
    
    def num_to_word_(self,cnt_list):
        #Function to convert numbers to words
        #Skip for now.
        pass
    
