import os
import sys
from itertools import chain
import argparse
import pandas as pd
from nltk.tag import pos_tag
from stanfordcorenlp import StanfordCoreNLP
import spacy
from flair.nn import Classifier
from flair.models import SequenceTagger
from flair.data import Sentence
import stanza

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from corpus import Corpus
from utils import evaluate
from config import Config


def nltk_pos_tag(identifiers, tags):
    print('nltk')
    total = len(identifiers)
    total_tokens = len(list(chain(*identifiers)))
    correct_ids = 0
    correct_tokens = 0
    out_tags = []
    for i, (identifier, tag) in enumerate(zip(identifiers, tags)):
        result = pos_tag(identifier)
        result_tags = [tag for _, tag in result]

        out_tags.append(result_tags)

        if tag == result_tags:
            correct_ids += 1

        for j, (golden, out) in enumerate(zip(tag, result_tags)):
            if golden == out:
                correct_tokens += 1
    print("ExactMatch: {}".format(correct_ids/total))
    print("Token Accuracy: {}".format(correct_tokens/total_tokens))
    return out_tags


def corenlp_pos_tag(identifiers, tags, path2stanford):
    nlp = StanfordCoreNLP(path2stanford)
    # print(nlp.pos_tag('Get Name'))
    total = len(identifiers)
    total_tokens = len(list(chain(*identifiers)))
    correct_ids = 0
    correct_tokens = 0
    out_tags = []
    wrong_tokenization = 0
    for i, (identifier, tag) in enumerate(zip(identifiers, tags)):
        stanford_input = " ".join(identifier)
        result = nlp.pos_tag(stanford_input)
        result_tags = [tag for _, tag in result]
        tokens = [token for token, _ in result]

        if tag == result_tags:
            correct_ids += 1

        if len(result_tags) == len(identifier):
            for j, (golden, out) in enumerate(zip(tag, result_tags)):
                if golden == out:
                    correct_tokens += 1
        elif len(result_tags) > len(tag):
            for k, token in enumerate(identifier):
                if token == tokens[k]:
                    if tag[k] == result_tags[k]:
                        correct_tokens += 1
                else:
                    for l in range(k+1, len(tokens)):
                        temp_token = ''.join(tokens[k:l+1])
                        if temp_token == token:
                            del tokens[k:l+1]
                            del result_tags[k:l+1]
                            tokens.insert(k, temp_token)
                            result_tags.insert(k, '<UNK>')
                            wrong_tokenization += 1
                            break
                        if ('+'+temp_token) == token:
                            # + and wrong number tokenization
                            del tokens[k:l+1]
                            del result_tags[k:l+1]
                            tokens.insert(k, '+'+temp_token)
                            result_tags.insert(k, '<UNK>')
                            wrong_tokenization += 1
                            break
                # except IndexError as e:
                #     print(k, identifier[k], tokens, len(tag), tag)
        elif len(result_tags) < len(tag):
            # nl tel number may be wrong tokenized "1 201 123" will be processed as one token
            for k, token in enumerate(tokens):
                if token == identifier[k]:
                    if tag[k] == result_tags[k]:
                        correct_tokens += 1
                elif identifier[k] == '+':
                    # + is missing in this situation
                    tokens.insert(k, '+')
                    result_tags.insert(k, '<UNK>')
                    wrong_tokenization += 1
                else:
                    for l in range(k+1, len(identifier)):
                        temp_token = ' '.join(identifier[k:l+1])
                        if temp_token == token:
                            del tokens[k]
                            del result_tags[k]
                            for m in range(k, l+1):
                                tokens.insert(m, identifier[m])
                                result_tags.insert(m, '<UNK>')
                                wrong_tokenization += 1
                            break
        if len(result_tags) != len(identifier):
            print('Wrong sequence: ', identifier, tag)
            print(tokens, result_tags)

        out_tags.append(result_tags)


    print('Stanford CoreNLP')
    print("ExactMatch: {}".format(correct_ids/total))
    print("Token Accuracy: {}".format(correct_tokens/total_tokens))
    return out_tags


def parse_opennlp_output(out_file_path):
    # Some opennlp output may 
    out_lines = []
    # evaluation
    start_line = 4
    end_line = -6
    # retrain
    # start_line = 3
    # end_line = -6


    with open(out_file_path, 'r') as f:
        lines = f.readlines()
        lines = lines[start_line:end_line]
        for line in lines:
            pairs = line.strip().split(' ')
            tag_line = []
            for pair in pairs:
                token_and_tag = pair.split('_')
                token = token_and_tag[0]
                tag = token_and_tag[-1]
                tag_line.append(tag)
            out_lines.append(tag_line)

    # print(len(out_lines), out_lines[-1])
    return out_lines

def opennlp_pos_tag(identifiers, tags, mode):
    opennlp_out = parse_opennlp_output(f'./opennlp_output/opennlp_{mode}_results.txt')
    if mode == 'nl':
        data = pd.read_csv('./NLData/nl_data.csv')
        # sentences = data['sentences'].values.tolist()
        poses = data['POS'].values.tolist()

        # sentences = [sentence.split(' ') for sentence in sentences]
        tags = [pos.split(' ') for pos in poses]
    # else:
    #     tags = corpus.test_tags
        # sentences = corpus.test_input
    em, token_acc = evaluate(opennlp_out, tags)
    print('OpenNLP')
    print("ExactMatch: {}".format(em))
    print("Token Accuracy: {}".format(token_acc))
    return opennlp_out

def spacy_pos_tag(identifiers, tags):
    nlp = spacy.load('en_core_web_sm')
    total = len(identifiers)
    total_tokens = len(list(chain(*identifiers)))
    correct_ids = 0
    correct_tokens = 0
    wrong_tokenization = 0
    out_tags = []
    for i, (identifier, tag) in enumerate(zip(identifiers, tags)):
        spacy_input = " ".join(identifier)
        result = nlp(spacy_input)
        result_tags = [token.tag_ for token in result]
        tokens = [token.text for token in result]

        if tag == result_tags:
            correct_ids += 1

        if len(result_tags) == len(identifier):
            for j, (golden, out) in enumerate(zip(tag, result_tags)):
                if golden == out:
                    correct_tokens += 1
        else:
            # id will be wrong tokenized as i d
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


        out_tags.append(result_tags)


        # if len(result_tags) != len(identifier):
        #     print(result_tags, identifier, spacy_input, [token for token in result])
        # if tag == result_tags:
        #     correct_ids += 1
        # for j, (golden, out) in enumerate(zip(tag, result_tags)):
        #     if golden == out:
        #         correct_tokens += 1
    print('Spacy')
    print("ExactMatch: {}".format(correct_ids/total))
    print("Token Accuracy: {}".format(correct_tokens/total_tokens))
    return out_tags


def flair_pos_tag(identifiers, tags):
    # tagger = SequenceTagger.load('pos')
    tagger = Classifier.load('/root/.flair/models/pos-english/pytorch_model.bin')
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
        # print(sentence.get_labels()[0].value)
        # break
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


        out_tags.append(result_tags)
        # if len(result_tags) != len(identifier):
        #     print(result_tags, identifier, flair_input, tag)
        #
        # if tag == result_tags:
        #     correct_ids += 1
        # for j, (golden, out) in enumerate(zip(tag, result_tags)):
        #     if golden == out:
        #         correct_tokens += 1
    print('Flair')
    print("ExactMatch: {}".format(correct_ids/total))
    print("Token Accuracy: {}".format(correct_tokens/total_tokens))
    return out_tags


def stanza_pos_tag(identifiers, tags):
    # stanza.download('en')
    # model_path = './saved_models/pos/en_test_charlm_tagger.pt'
    # nlp = stanza.Pipeline('en', processors='tokenize,mwt,pos', pos_model_path=model_path, download_method=None)
    nlp = stanza.Pipeline('en', processors='tokenize,mwt,pos', download_method=None)
    total = len(identifiers)
    total_tokens = len(list(chain(*identifiers)))
    correct_ids = 0
    correct_tokens = 0
    out_tags = []
    wrong_tokenization = 0
    for i, (identifier, tag) in enumerate(zip(identifiers, tags)):
        stanford_input = " ".join(identifier)
        result = nlp(stanford_input)
        result_tags = []
        tokens = []
        for sent in result.sentences:
            result_tags =  [word.xpos for word in sent.words]
            tokens = [word.text for word in sent.words]

        if tag == result_tags:
            correct_ids += 1

        if len(result_tags) == len(identifier):
            for j, (golden, out) in enumerate(zip(tag, result_tags)):
                if golden == out:
                    correct_tokens += 1
        elif len(result_tags) > len(tag):
            for k, token in enumerate(identifier):
                if token == tokens[k]:
                    if tag[k] == result_tags[k]:
                        correct_tokens += 1
                else:
                    for l in range(k+1, len(tokens)):
                        temp_token = ''.join(tokens[k:l+1])
                        if temp_token == token:
                            del tokens[k:l+1]
                            del result_tags[k:l+1]
                            tokens.insert(k, temp_token)
                            result_tags.insert(k, '<UNK>')
                            wrong_tokenization += 1
                            break
                        if ('+'+temp_token) == token:
                            # + and wrong number tokenization
                            del tokens[k:l+1]
                            del result_tags[k:l+1]
                            tokens.insert(k, '+'+temp_token)
                            result_tags.insert(k, '<UNK>')
                            wrong_tokenization += 1
                            break
                # except IndexError as e:
                #     print(k, identifier[k], tokens, len(tag), tag)
        elif len(result_tags) < len(tag):
            # nl tel number may be wrong tokenized "1 201 123" will be processed as one token
            for k, token in enumerate(tokens):
                if token == identifier[k]:
                    if tag[k] == result_tags[k]:
                        correct_tokens += 1
                elif identifier[k] == '+':
                    # + is missing in this situation
                    tokens.insert(k, '+')
                    result_tags.insert(k, '<UNK>')
                    wrong_tokenization += 1
                else:
                    for l in range(k+1, len(identifier)):
                        temp_token = ' '.join(identifier[k:l+1])
                        if temp_token == token:
                            del tokens[k]
                            del result_tags[k]
                            for m in range(k, l+1):
                                tokens.insert(m, identifier[m])
                                result_tags.insert(m, '<UNK>')
                                wrong_tokenization += 1
                            break
        if len(result_tags) != len(identifier):
            print('Wrong sequence: ', identifier, tag)
            print(tokens, result_tags)

        out_tags.append(result_tags)


    print('Stanford Stanza')
    print("Identifier Accuracy: {}".format(correct_ids/total), correct_ids, total)
    print("Token Accuracy: {}".format(correct_tokens/total_tokens), correct_tokens, total_tokens)
    return out_tags

def list2str(lists):
    new_lists = []
    for l in lists:
        new_lists.append(' '.join(l))
    return new_lists


if __name__ == '__main__':
    # nltk.download('averaged_perceptron_tagger')
    # parser = argparse.ArgumentParser(description='Evaluation on six natural language taggers')
    # mode id or nl
    # parser.add_argument('-m', '--mode', help='Choose method/args/class/all or nl to process.')
    # args = parser.parse_args()
    config = Config()
    eva_mode = config.mode
    data_path = config.data_path
    print(eva_mode)

    if eva_mode == 'nl':
        data = pd.read_csv(data_path+'/NLData/nl_data.csv')
        sequences = data['SEQUENCE'].values.tolist()
        poses = data['POS'].values.tolist()

        sequences = [sentence.split(' ') for sentence in sequences]
        poses = [pos.split(' ') for pos in poses]
    else:
        corpus = Corpus(data_path, eva_mode)
        sequences = corpus.test_input
        poses = corpus.test_tags

    nltk_out = nltk_pos_tag(sequences, poses)
    corenlp_out = corenlp_pos_tag(sequences, poses, path2stanford='/home/stanford-corenlp')
    opennlp_out = opennlp_pos_tag(sequences, poses, mode=eva_mode)
    spacy_out = spacy_pos_tag(sequences, poses)
    flair_out = flair_pos_tag(sequences, poses)
    stanza_out = stanza_pos_tag(sequences, poses)
    # flair_pos_tag()
    data = {'SEQUENCE': list2str(sequences),
            'POS': list2str(poses),
            'NLTK': list2str(nltk_out),
            'CORENLP': list2str(corenlp_out),
            'OPENNLP': list2str(opennlp_out),
            'SPACY': list2str(spacy_out),
            'FLAIR': list2str(flair_out),
            'STANZA': list2str(stanza_out)}
    df = pd.DataFrame(data)
    df.to_csv(f'./evaluation_{eva_mode}.csv')
