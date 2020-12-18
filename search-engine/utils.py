from IPython.display import display

def display_result(df, sim_scores, pop_scores, query, show = 20):
    results = df.merge(sim_scores.merge(pop_scores, left_on=['doc_index'], right_on=['id'], left_index=True), on=['id'], left_index=True)
    results['total_score'] = 0.7*results['similarity']+0.3*results['pop_score']
    display(results.sort_values(by=['total_score'], ascending=False).reset_index().head(show)[['total_score', 'text']])