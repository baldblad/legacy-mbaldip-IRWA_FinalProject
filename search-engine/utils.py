'''Module used to store the functions to display the rankings and any other needed function.'''
from IPython.display import display
import os

clear = lambda: os.system('clear')

def display_result(df, sim_scores, pop_scores, query, customScore, show = 20):
    '''Interactive display on the scores obtained for each document given a query.
    Final score is computed using the custom score or not depending on 'pop_scores' argument.
    Ranking is performed before display.'''
    results = df.merge(sim_scores.merge(pop_scores, left_on=['doc_index'], right_on=['id'], left_index=True), on=['id'], left_index=True)
    if customScore == 'y':
        results['total_score'] = 0.7*results['similarity']+0.3*results['pop_score']
    else:
        results['total_score'] = results['similarity']

    # Perform ranking using score. The dataframe index is used as a rank of the instance.
    results = results.sort_values(by=['total_score'], ascending=False).reset_index().head(show)[['total_score', 'text']]
    
    # Display and expore the results interactively
    display(results)
    explore_text = input('\nIntroduce the document index (number) to view the full text, use \'q\' to go back: ')
    clear()
    while (explore_text!='q'):
        try:
            print(results['text'].iloc[int(explore_text)])
        except:
            print('Wrong formating of the document index.')
        
        input('\nPress any key to go continue')
        display(results)
        explore_text = input('\nIntroduce the document index (number) to view the full text, use \'q\' to go back: ')
        clear()