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

from spellchecker import spell_checker
from contractmapping import contraction_mapping

stop_words = stopwords.words('english')

class NLP_Preprocessor:
    def __init__(self,df,
                 lowercase=True,
                 rm_punctuations=True,
                 rm_stopwords=True,
                 rm_pr_stopword_list = [],
                 rm_freq_words=False,
                 rm_rare_words= False,
                 rm_emoji=True,
                 rm_emoticons=False,
                 conv_emoticons_to_word = False,
                 conv_emoji_to_words = False,
                 rm_urls=True,
                 rm_html_tags=False,
                 chat_word_conv=False,
                 spell_correct=True,
                 num_to_word=True,
                 exp_contraction=True,
                 word_token=True,
                 sent_token=False,
                 lemma=True,
                 stem=False,
                ):
    
        self.df = df
        self.lowercase_fg = lowercase
        self.punc_fg = rm_punctuations
        self.stopwords_fg = rm_stopwords
        self.pr_stopwords_list = rm_pr_stopword_list #Data wise stop word list given by the data scientist.
        self.freq_words_fg = rm_freq_words
        self.rare_words_fg = rm_rare_words
        self.emoji_fg = rm_emoji
        self.emoticons_fg = rm_emoticons
        self.emoticon_to_word_fg = conv_emoticons_to_word
        self.emoji_to_word_fg = conv_emoji_to_words
        self.url_fg = rm_urls
        self.html_tag_fg = rm_html_tags
        self.chat_word_conv_fg = chat_word_conv
        self.spell_correct_fg = spell_correct
        self.num_to_word_fg = num_to_word
        self.exp_contraction_fg = exp_contraction
        self.word_token_fg = word_token
        self.sent_token = sent_token
        self.lemma = lemma
        self.stem = stem
        self.fdist = FreqDist()
    
    def lowercase(self,
                   processing_col,
                   new_col):
       
        self.df[new_col] = self.df[processing_col].str.lower()
    

    def url_removal(self,
                     processing_col,
                     new_col):
        
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        self.df[new_col] = self.df[processing_col].apply(lambda text: url_pattern.sub(r'',text))


    def html_tag(self,
                  processing_col,
                  new_col):
       
        html_pattern = re.compile('<.*?>')
        self.df[new_col] = self.df[processing_col].apply(lambda text: html_pattern.sub(r'',text))
    
    
    def rm_punc(self,
                processing_col,
                new_col,
                punct_remove='!"#$%&\'()*+,-./:;<=>?@[\\]^_{|}~\n`'):

        self.df[new_col] = self.df[processing_col].apply(lambda text: text.translate(str.maketrans('','',punct_remove)))
    
    
    def rm_stopwords(self,
                      processing_col,
                      new_col,
                      stopword_list=[],):
        stop_words = set(stopwords.words('english'))
        #if stopword_list and add=True:
            #Adding both the stop words
        
        self.df[new_col] = self.df[processing_col].apply(lambda text: ' '.join(word for word in text.split() if word not in stop_words))
        self.df[new_col] = self.df[processing_col].apply(lambda text: text.split(' '))
    

    def word_token(self,
                    processing_col,
                    new_col):
        #Function to convert all the data into tokens.
        self.df[new_col] = self.df[processing_col].apply(lambda text: nltk.tokenize.word_tokenize(text))
    
    
    def sent_tokens(self,
                     processing_col,
                     new_col):
        #Function to convert all the data into sentence tokens
        self.df[new_col] = self.df[processing_col].apply(lambda text: nltk.tokenize.sent_tokenize(text))
    
    
    def lemmatiz(self,
               processing_col,
               new_col):
        #Function to apply lemmatization
        lemmatizer = WordNetLemmatizer() 
        self.df[new_col] = self.df[processing_col].apply(lambda tokens : " ".join(lemmatizer.lemmatize(token) for token in tokens ))
        
    
    def stemmer(self,
                 processing_col,
                 new_col):
        #Function to apply stemmer
        ps = PorterStemmer()
        self.df[new_col] = self.df[processing_col].apply(lambda tokens : " ".join(ps.stem(token) for token in tokens))
                
    
    def spell_correct_(self,):
        #Function to check the spelling.
        self.df['spell_corrected'] = self.df[self.col_name].apply(lambda token_list : spell_checker(token_list))


    def expand_the_contraction(self):
        #Function to expand the contractions.
        self.df['expanded'] = self.df[self.col_name].apply(lambda tokens: " ".join(contraction_mapping(token) for token in tokens))


    def freq_words(self,
                  processing_col):
        for token_list in list(self.df[processing_col]):
            for token in token_list:
                self.fdist[token] += 1
    
    
    def rm_freq_rare_words_(self,
                            preprocessing_col,
                            new_col_freq_words='top_freq_word',
                            new_col_rare_words='top_rare_words',
                            top_freq_percent=10,
                            top_rare_percent=1):
        #Function to remove frequent word or rare word depending upon the flag
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