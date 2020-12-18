from IPython.display import display
import os

clear = lambda: os.system('clear')

def display_result(df, sim_scores, pop_scores, query, customScore, show = 20):
    results = df.merge(sim_scores.merge(pop_scores, left_on=['doc_index'], right_on=['id'], left_index=True), on=['id'], left_index=True)
    if customScore == 'y':
        results['total_score'] = 0.7*results['similarity']+0.3*results['pop_score']
    else:
        results['total_score'] = results['similarity']
    results = results.sort_values(by=['total_score'], ascending=False).reset_index().head(show)[['total_score', 'text']]
    
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