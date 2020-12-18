# GENERAL IMPORTS #
import pandas as pd
import numpy as np
from IPython.display import display
import os

# custom modules imports:
import dataCleaning as dC
import ranking
import utils

# Not needed if using a database
'''# IMPORTS: Twitter Scrapping #
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
from tweepy import API
from tweepy import Cursor
import json'''

# PATHS MANAGING
cwd = os.getcwd() # get current working directory
SEpath = './search-engine' # path to the search engine code

# global functions
clear = lambda: os.system('clear')

def main():

    # read tweets database
    global df
    try:
        df = pd.read_csv(os.path.join(SEpath,"database/df_tweets_clean.csv") )
    except:
        print("Database is missing or unaccessible.")
        print("Shuting down the search engine.")
        return
    # precautions:
    df = df.dropna()

    # Build the inverted index:
    inverted_index = ranking.create_inverted_index(df)

    # start interactive
    clear()
    while True:
        print('\n\nWELCOME TO THE US ELECTIONS SEARCH ENGINE\n')
        query = input('What do you want to know about? ')

        # Clean the query to process:
        cquery = dC.preprocess_text(query)
        if len(cquery.split())==0:
            print('\'{}\' is too generic. Please, add more words to your query.'.format(query))
        else:
            # ranking:
            out = ranking.runDocumentsScores(inverted_index, cquery, df)
            if out:
                dfq, pop_scores, sim_scores = out

                # display the results
                utils.display_result(dfq, sim_scores, pop_scores, query)

        # ask quit condition and clean:
        answ = input('\n\nDo you want to search again?(y/n) ')
        clear()
        if answ == 'n':
            break
        

    print('\nThanks for using our search engine!')

    return


main()