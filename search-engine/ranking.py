import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def create_inverted_index(df):   
    inverted_index = {}
    for _, tweet in df.iterrows():
        for word in tweet['cleaned_text'].split():
            if word in inverted_index:
                inverted_index[word].append(tweet['id'])
            else:
                inverted_index[word] = [tweet['id']]
    return inverted_index

def filter_index(inverted_index, queue):
    tweets_to_show = set()
    for word in queue.split():
        try:
            if len(tweets_to_show)==0:
                tweets_to_show = set(inverted_index[word])
            elif word in inverted_index:
                docs = set(inverted_index[word])
                tweets_to_show = tweets_to_show.intersection(docs)
        except KeyError:
            print("Word \'{}\' is not in the corpus".format(word))
    return list(tweets_to_show)

def computeTfidfSimilarities(tweets_to_show, query, df):
    
    tweets_to_rank = df[df['id'].isin(tweets_to_show)]['cleaned_text']
    X = tweets_to_rank.append(pd.Series(query), ignore_index=True)
    
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(X)
    TfidfMatrix = pd.DataFrame.sparse.from_spmatrix(X)
    TfidfMatrix.rename(columns={val:key for (key,val) in vectorizer.vocabulary_.items()}, inplace=True)
    query = TfidfMatrix.tail(1)
    TfidfMatrix.drop(TfidfMatrix.tail(1).index, inplace=True)
    query = query.to_numpy().reshape(1,-1)
    similarities = cosine_similarity(TfidfMatrix.to_numpy(), query)
    
    doc_simi = pd.DataFrame(data=similarities, 
                            index=df[df['id'].isin(tweets_to_show)]['id'],
                            columns=['similarity'])
    
    doc_simi['doc_index'] = doc_simi.index
    return doc_simi

def computePopularityScore(tweets_to_show, df):
    dfn = df[df['id'].isin(tweets_to_show)][['id','quote_count','reply_count','retweet_count','favorite_count']]
    dfn['pop_score'] = 0.5*(dfn['quote_count']+dfn['retweet_count'])+0.3*dfn['favorite_count']+0.2*dfn['reply_count']
    dfn['pop_score'] = dfn['pop_score']/max(dfn['pop_score'].values)
    return dfn.drop(columns=['retweet_count','favorite_count'])

def runDocumentsScores(inverted_index, query, df):
    # filter documents by query using the inverted index
    tweets_to_show = filter_index(inverted_index, query)
    if len(tweets_to_show)==0:
        print('No results found :(')
        return None

    # compute the query-doc TF-IDF similarities
    simis = computeTfidfSimilarities(tweets_to_show, query, df)

    # Compute the custom score (indicator of tweet popularity based on its attributes)
    popu = computePopularityScore(tweets_to_show, df)

    return df, popu, simis