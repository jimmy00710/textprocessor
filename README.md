# textprocessor
A script which performs almost all types of preprocessing on text. data

Scrips supports following function - 


> Lowercase <br>
> URL Removal <br>
> HTML Tag <br>
> Remove Punctuation <br>
> Remove Stopwords <br>
> Word Token <br>
> Sent Token <br>
> Lemmatize <br>
> Stemmer <br>
> Spell Correct <br>
> Expand the Contraction <br>
> Remove Freq Words <br>
> Remove Rare Words <br>
> Remove Emoji and Emoticons <br>
> Convert Emoji <br>
> Chat conversion <br>
> Number to words <br>


#### <i>  All the functions have the functionality to do in the same column or create a new column. i.e inplace = True or Newcolumn Name - </i>  
### Lowercase - 
In lowercase function we lowercase the whole column. We have given functionality to do lowercase on the same column or if we want to create a new column in case.

## URL Removal - 
Similarly URL removal removes the URL.

## HTML Tag - 
This function removes the html tag.

## Punctuation Removal - 
This function removes the punctuation.

## Word Token - 
Once the data is cleaned we can tokenize the words.

## Sent Token - 
Tokenize the data in sentences.

## Lemmatize - 
Use lemmatization on the tokenize data.

## Stemmer - 
Use stemming on the tokenize data or lemmatize data. 

## Spell Correct - 
It checks if the spelling is correct or not. In this I am using a library.

## Expand the contraction - 
In this we have kept a mapping of all the contractions. We can create a mapping of our own contraction too. Based on the key we can map the contraction to exansion. 

## Remove Freq Words - 
This function find the top freq words and remove them. It takes the input of top n% and remove them. 

## Remove Rare Words -
This function find the rares words and remove them. It takes the input of n% and remove it. 

## Remove Emoji and Emoticons - 
This function removes emojis and emoticons. 

## Convert Emoji 
In case we don't want to remove emoji but we want to convert them into text data we can use it. (Feature need to Add)

## 