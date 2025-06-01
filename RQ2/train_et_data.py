from itertools import chain
from nltk.tag import PerceptronTagger
import pandas as pd
from flair.embeddings import WordEmbeddings, StackedEmbeddings, FlairEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from flair.nn import Classifier
from flair.data import Corpus, Sentence
from flair.datasets import ColumnCorpus
import spacy
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from process_data import create_stanford_data, preprocess2spacy, read_stanford
from tag import nltk_pos_tag, spacy_pos_tag, stanza_pos_tag, flair_pos_tag, corenlp_pos_tag


def create_flair_data(data, mode='train'):
    with open('../dataset/flair_format/et_data_flair_format/et_flair_'+mode+'.txt', 'w') as file:
        for id in data:
            for word, tag in id:
                line = word + ' ' + tag + '\n'
                file.write(line)
            file.write('\n')


def train_flair(train_data, test_data):
    # data
    training_data = []
    # create_flair_data(train_data, mode='train')
    # create_flair_data(test_data, 'test')

    columns = {0: 'text', 1: 'pos'}
    data_folder = '../dataset/flair_format/et_data_flair_format/'
    corpus: Corpus = ColumnCorpus(data_folder, columns,
                                  train_file='et_flair_train.txt',
                                  # dev_file='flair_dev.txt',
                                  test_file='et_flair_test.txt'
                                  )
    # test_file='test.txt',
    # dev_file='dev.txt'

    label_type = 'pos'
    embedding_types = [
        FlairEmbeddings('news-forward'),
        FlairEmbeddings('news-backward'),
    ]
    embeddings = StackedEmbeddings(embeddings=embedding_types)
    # default_tagger = Classifier.load('pos')
    label_dict = corpus.make_label_dictionary(label_type=label_type)
    tagger = SequenceTagger(hidden_size=256,
                            embeddings=embeddings,
                            tag_dictionary=label_dict,
                            tag_type=label_type,
                            use_crf=True)
    trainer = ModelTrainer(tagger, corpus)

    trainer.train('./et_flair_tagger',
                  learning_rate=0.1,
                  mini_batch_size=32)

    # sentence = Sentence('get name')
    # print(tagger.predict(sentence))
    return tagger


def eva_flair(identifiers, tags):
    tagger = Classifier.load('../model/flair/et_flair_tagger/best-model.pt')
    print(tagger)
    total = len(identifiers)
    total_tokens = len(list(chain(*identifiers)))
    correct_ids = 0
    correct_tokens = 0
    wrong_tokenization = 0
    out_tags = []
    for i, (identifier, tag) in enumerate(zip(identifiers, tags)):
        flair_input = " ".join(identifier)
        # print(flair_input)
        sentence = Sentence(flair_input)
        tagger.predict(sentence)

        result_tags = [token.value for token in sentence.get_labels()]
        tokens = [token.text for token in sentence.tokens]

        if tag == result_tags:
            correct_ids += 1

        if len(result_tags) == len(identifier):
            for j, (golden, out) in enumerate(zip(tag, result_tags)):
                if golden == out:
                    correct_tokens += 1
        else:
            for k, token in enumerate(identifier):
                if token == tokens[k]:
                    if tag[k] == result_tags[k]:
                        correct_tokens += 1
                else:
                    for l in range(k+1, len(tokens)):
                        temp_token = ''.join(tokens[k:l+1])
                        # print(temp_token)
                        if temp_token == token:
                            del tokens[k:l+1]
                            del result_tags[k:l+1]
                            tokens.insert(k, temp_token)
                            result_tags.insert(k, '<UNK>')
                            wrong_tokenization += 1
                            break
        if len(result_tags) != len(identifier):
            print('Wrong sequence: ', identifier, tag)
            print(tokens, result_tags)


        out_tags.append(" ".join(result_tags))

    print('Flair')
    print("Identifier Accuracy: {}".format(correct_ids/total))
    print("Token Accuracy: {}".format(correct_tokens/total_tokens))
    return out_tags


def train_nltk(train_path):
    print('nltk:')
    print("perceptron model")
    # [[('the', ''),
    training_data = read_et_data(train_path)

    trained_tagger = PerceptronTagger()
    trained_tagger.train(training_data, './retrain.et.tagger.pickle')

    # print(trained_tagger.evaluate(training_data))
    return trained_tagger


def eva_nltk(testing_data, model_name='retrain.et.tagger'):
    print('eva-mode: nltk:')
    print("perceptron model")
    # brown_tagged_sents = brown.tagged_sents()
    # print(brown_tagged_sents)
    # [[('the', ''),
    # testing_data = read_et_data(testing_path)

    trained_tagger = PerceptronTagger()
    trained_tagger.load('../model/nltk/'+model_name+'.pickle')
    # trained_tagger.train(training_data, 'retrain.per.tagger')

    print('Acc: ')
    print(trained_tagger.evaluate(testing_data))

    out_tag = []
    input_data = []
    for line in testing_data:
        # print(line)
        line_input = []
        tag_output = []
        for token, tag in line:
            line_input.append(token)
        line_str = ' '.join(line_input)
        input_data.append(line_str)
        line_out = trained_tagger.tag(line_input)
        for token, tag in line_out:
            tag_output.append(tag)
        tag_str = ' '.join(tag_output)
        out_tag.append(tag_str)
    # print(testing_data)
    # print(out_tag)
    return out_tag


def read_et_data(path):
    df = pd.read_csv(path)
    words = df['WORD'].values.tolist()
    tags = df['CORRECT_TAG'].values.tolist()
    position = df['POSITION'].values.tolist()
    max_position = df['MAXPOSITION'].values.tolist()
    # print(words, tags)
    pairs = []
    id_temp = []
    for i, (word, tag) in enumerate(zip(words, tags)):
        pair = (str(word).lower(), tag)
        if position[i] == 0 and len(id_temp)>0:
            pairs.append(id_temp)
            id_temp = []
        id_temp.append(pair)
    pairs.append(id_temp)

    # print(pairs)
    return pairs


def process_pair_to_list(data):
    identifiers, poses = [], []
    for line in data:
        identifier, pos_line = [], []
        for (token, pos) in line:
            # print(token)
            identifier.append(token)
            pos_line.append(pos)
        identifiers.append(identifier)
        poses.append(pos_line)
    return identifiers, poses


def map2et(tag_lines):
    map_df = pd.read_csv('./map.csv')
    map_dict = map_df.set_index('ORI')['ET'].to_dict()
    mapped_tag_lines = []
    for line in tag_lines:
        mapped_line = []
        for ori_tag in line:
            if ori_tag not in map_dict.keys():
                mapped_tag = 'UNK'
            else:
                mapped_tag = map_dict[ori_tag]
            mapped_line.append(mapped_tag)
        mapped_tag_lines.append(" ".join(mapped_line))
    return mapped_tag_lines


def cal_acc_em(outs, true_labels):
    true_item_number = 0
    item_number = len(true_labels)
    true_token_number = 0
    token_number = 0
    for out, labels in zip(outs, true_labels):
        if out == labels:
            true_item_number += 1
        out_list = out.split(' ')
        label_list = labels.split(' ')
        for i in range(len(label_list)):
            token_number += 1
            if i >= len(out_list):
                continue
            if out_list[i] == label_list[i]:
                true_token_number += 1
    token_accuracy = true_token_number/token_number
    em = true_item_number/item_number

    print("Token Accracy: %f, %d/%d "%(token_accuracy, true_token_number, token_number))
    print("Exact Match: %f, %d/%d "%(em, true_item_number, item_number))


def test_spacy(ids, poses, model_path='../model/spacy/et_retrain/model-best'):
    nlp = spacy.load(model_path)
    out_list = []
    for id, pos in zip(ids, poses):
        doc = nlp(" ".join(id))
        tags = []
        for word in doc:
            tags.append(word.tag_)
        out_list.append(' '.join(tags))
    return out_list



if __name__ == '__main__':
    training_path = '../dataset/ensemble_tagger_training_data/training_data.csv'
    testing_path = '../dataset/ensemble_tagger_training_data/unseen_testing_data.csv'
    train_data = read_et_data(training_path)
    test_data = read_et_data(testing_path)
    test_identifiers, test_labels = process_pair_to_list(test_data)
    train_identifiers, train_labels = process_pair_to_list(train_data)
    test_labels_str = [' '.join(line) for line in test_labels]
    
    # NLTK
    # train_nltk(training_path)
    nltk_out = eva_nltk(test_data)
    cal_acc_em(nltk_out, test_labels_str)

    # flair
    # train_flair(train_data, test_data)
    flair_out = eva_flair(test_identifiers, test_labels)
    cal_acc_em(flair_out, test_labels_str)

    
    # corenlp
    # create_stanford_data(train_identifiers, train_labels, 'etdata_stanford_train.txt')
    # create_stanford_data(test_identifiers, test_labels, 'etdata_stanford_test.txt')
    print('CoreNLP:')
    corenlp_out = read_stanford('./stanford_et_out/etdata_stanford_out.txt')
    cal_acc_em(corenlp_out, test_labels_str)

    # spacy
    # preprocess2spacy(train_identifiers, train_labels, data_type='et_train')
    # preprocess2spacy(test_identifiers, test_labels, data_type='et_test')
    print('spaCy:')
    spacy_et_outs = test_spacy(test_identifiers, test_labels_str)
    cal_acc_em(spacy_et_outs, test_labels_str)

    out_ids = [' '.join(line) for line in test_identifiers]
    data_dict = {'SEQUENCE': out_ids, 'POS':test_labels_str, 'NLTK': nltk_out, 'CoreNLP': corenlp_out, 'spaCy': spacy_et_outs, 'Flair': flair_out}
    df = pd.DataFrame(data_dict)
    df.to_csv('./et_evaluation.csv')