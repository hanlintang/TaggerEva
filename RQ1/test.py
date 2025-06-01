import pandas as pd
from utils import evaluate, str2list

modes = ['all', 'method', 'args', 'class', 'nl']
# fit evaluation_xx.csv

for mode in modes:
    path = f'../evaluation_results/evaluation_{mode}.csv'
    df = pd.read_csv(path)
    nltk_out = str2list(df['NLTK'].values.tolist())
    corenlp_out = str2list(df['CORENLP'].values.tolist())
    opennlp_out = str2list(df['OPENNLP'].values.tolist())
    spacy_out = str2list(df['SPACY'].values.tolist())
    flair_out = str2list(df['FLAIR'].values.tolist())
    stanza_out = str2list(df['STANZA'].values.tolist())

    pos_list = str2list(df['POS'].values.tolist())
    print(mode)
    print('nltk')
    evaluate(nltk_out, pos_list)

    print('corenlp')
    evaluate(corenlp_out, pos_list)
    
    print('opennlp')
    evaluate(opennlp_out, pos_list)

    print('spacy')
    evaluate(spacy_out, pos_list)

    print('flair')
    evaluate(flair_out, pos_list)

    print('stanza')
    evaluate(stanza_out, pos_list)