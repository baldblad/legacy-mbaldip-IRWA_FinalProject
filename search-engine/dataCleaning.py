'''Module that contains the functions used to clean text, mainly the query and originally the dataset.
Although the dataset was cleaned with the same functions, the data used by this software is already cleaned
and ready to use (see database folder).
'''
import nltk
from nltk.tokenize import TweetTokenizer
from nltk.stem.snowball import SnowballStemmer

nltk.download("stopwords", quiet=True)

# Functions for preprocessing text
def tokenizeTweet(text):
    '''Convert string query into list of tokens.'''
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
    return tokenizer.tokenize(text)

def convertHashtags(tokens):
    '''Remove hashtags and keep the text from the hashtag'''
    return [word.replace("#",'') for word in tokens]

def dropPunctuation(tokens):
    '''Drop the punctuation signs'''
    return [word for word in tokens if word.isalpha()]

def removeStopWords(tokens):
    '''Removes stopwords; common and with small semantic charge'''
    return [word for word in tokens if word not in nltk.corpus.stopwords.words('english')]

def stemWords(tokens):
    '''Perform stemming to a list of tokens'''
    stemmer = SnowballStemmer("english")
    return [stemmer.stem(word) for word in tokens]

def listToString(tokens):
    '''Convert a list of tokens (back) to a string'''
    return ' '.join(tokens)

def preprocess_text(text):
    '''Master function that calls the needed functions to clean the query
    in a way that it can be compared with the processed documents'''
    
    text = tokenizeTweet(text)
    text = convertHashtags(text)
    text = dropPunctuation(text)
    text = removeStopWords(text)
    text = stemWords(text)
    text = listToString(text)

    return text