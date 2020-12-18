import nltk
from nltk.tokenize import TweetTokenizer
from nltk.stem.snowball import SnowballStemmer

nltk.download("stopwords", quiet=True)

# Functions for preprocessing text
def tokenizeTweet(text):
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
    return tokenizer.tokenize(text)

def convertHashtags(tokens):
    return [word.replace("#",'') for word in tokens]

def dropPunctuation(tokens):
    return [word for word in tokens if word.isalpha()]

def removeStopWords(tokens):
    return [word for word in tokens if word not in nltk.corpus.stopwords.words('english')]

def stemWords(tokens):
    stemmer = SnowballStemmer("english")
    return [stemmer.stem(word) for word in tokens]

def listToString(tokens):
    return ' '.join(tokens)

def preprocess_text(text):
    
    text = tokenizeTweet(text)
    text = convertHashtags(text)
    text = dropPunctuation(text)
    text = removeStopWords(text)
    text = stemWords(text)
    text = listToString(text)

    return text