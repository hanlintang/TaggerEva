import pandas as pd
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import Config
from corpus import Corpus


def create_et_data(config, data_path, mode):
    columns = ['ID', 'TYPE', 'DECLARATION', 'PROJECT', 'FILE']
    corpus = Corpus(data_path, mode=config.mode)
    # print(corpus.test_data.columns)
    length = len(corpus.test_tags)

    # id
    id_list = list(range(length))
    # type
    type = []
    if mode == 'class':
        type = ['class'] * length
    elif mode == 'args':
        type = corpus.test_data['type'].values.tolist()
    print(corpus.test_data.columns)
    # declaration
    declaration = corpus.test_data['SEQUENCE'].values.tolist()
    #project
    project = corpus.test_data['PROJECT'].values.tolist()
    # file
    file = corpus.test_data['FILE'].values.tolist()

    data = {'ID': id_list, 'TYPE': type, 'DECLARATION': declaration, 'PROJECT': project, 'FILE':file}
    df = pd.DataFrame(data)
    df.to_csv(f'./{mode}_et_input.csv')
    return df


if __name__ == '__main__':
    config = Config()
    data = create_et_data(config, config.data_path, mode='method')
    print(data)