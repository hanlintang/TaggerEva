import os
import pandas as pd



class Corpus:
    def __init__(self, data_path, mode='method'):
        self.train_data = self.get_data(os.path.join(data_path, 'MNTrain'), 'train.csv')
        self.dev_data = self.get_data(os.path.join(data_path, 'MNTrain'), 'dev.csv')
        self.test_data = self.get_test_data(os.path.join(data_path, 'IDData'), mode)


        self.train_input, self.train_tags = self.get_words_and_tags(self.train_data)
        if self.dev_data is not None:
            self.dev_input, self.dev_tags = self.get_words_and_tags(self.dev_data)
        self.test_input, self.test_tags = self.get_words_and_tags(self.test_data)


    def get_words_and_tags(self, data):
        identifiers = data['SEQUENCE'].values.tolist()
        identifiers = [identifier.split(' ') for identifier in identifiers]
        tags = data['POS'].values.tolist()
        tags = [tag.split(' ') for tag in tags]
        return identifiers, tags

    def get_data(self, data_path, file_name):
        path = os.path.join(data_path, file_name)
        if not os.path.exists(path):
            return None
        df = pd.read_csv(path)
        df['SEQUENCE'] = df['SEQUENCE'].str.lower()
        return df

    def get_test_data(self, path, mode='all'):
        arg_data = self.get_data(path, 'pdata.csv')
        method_data = self.get_data(path, 'mdata.csv')
        classname_data = self.get_data(path, 'cdata.csv')
        if mode=='class':
            return classname_data
        elif mode=='args':
            return arg_data
        elif mode=='method':
            return method_data
        else:
            return pd.concat([arg_data, method_data, classname_data])


if __name__ == "__main__":
    data_path = './dataset/'
    corpus = Corpus(data_path, 'class')
    print(len(corpus.test_tags))
