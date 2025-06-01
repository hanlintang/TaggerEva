import argparse
from itertools import chain

from nltk.tag import PerceptronTagger
from nltk.corpus import wordnet as wn
import spacy
from spacy.tokens import Doc, DocBin
from spacy.training import Example
import pandas as pd
from flair.embeddings import WordEmbeddings, StackedEmbeddings, FlairEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from flair.nn import Classifier
from flair.data import Corpus, Sentence
from flair.datasets import ColumnCorpus
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from corpus import Corpus
from utils import list2str, evaluate, str2list
from config import Config


def parse_opennlp_output(out_file_path):
    out_lines = []
    # evaluation
    # start_line = 4
    # end_line = -6
    # retrain
    start_line = 3
    end_line = -6
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

def train_nltk(train_ids, train_poses):
    print('nltk:')
    print("perceptron model")
    # brown_tagged_sents = brown.tagged_sents()
    # print(brown_tagged_sents)
    # [[('the', ''),
    training_data = []
    for id, pos in zip(train_ids, train_poses):
        pairs = [(word, tag) for word, tag in zip(id, pos)]
        training_data.append(pairs)

    # print(brown_tagger.evaluate(test_data))
    trained_tagger = PerceptronTagger()
    trained_tagger.train(training_data, save_loc='retrain.per.tagger.pickle')

    print(trained_tagger.accuracy(training_data))
    return trained_tagger

def eva_nltk(ids, poses, model_name='retrain.per.tagger'):
    print('eva-mode: nltk:')
    print("perceptron model")
    # brown_tagged_sents = brown.tagged_sents()
    # print(brown_tagged_sents)
    # [[('the', ''),
    testing_data = []
    for id, pos in zip(ids, poses):
        pairs = [(word, tag) for word, tag in zip(id, pos)]
        testing_data.append(pairs)

    # print(brown_tagger.evaluate(test_data))
    trained_tagger = PerceptronTagger()
    trained_tagger.load('../model/nltk/'+model_name+'.pickle')
    # trained_tagger.train(training_data, 'retrain.per.tagger')

    print('Acc: ')
    print(trained_tagger.accuracy(testing_data))

    out_tag = []
    for id in ids:
        pos = trained_tagger.tag(id)
        tags = [tag for _, tag in pos]
        out_tag.append(" ".join(tags))
    return out_tag



    # return trained_tagger

def read_stanford(path='stanford_out.txt'):
    out = []
    with open(path, 'r') as file:
        stanford_out = file.readlines()
        for line in stanford_out:
            pairs = line.strip().split(' ')
            tags = [pair.split('/')[-1] for pair in pairs]
            out.append(" ".join(tags))
    return out

def create_stanford_data(ids, poses, out_name='stanford_train.txt'):
    with open(out_name, 'w') as file:
        for id, tags in zip(ids, poses):
            line = []
            for word, tag in zip(id, tags):
                line.append(word+'/'+tag)
            out_line = ' '.join(line) + '\n'
            file.write(out_line)

def preprocess2spacy(ids, poses, data_type='train'):
    nlp = spacy.blank('en')
    data = []
    db = DocBin()
    for i, (id, pos) in enumerate(zip(ids, poses)):

        words = id
        tags = pos
        nlp(" ".join(id))
        try:
            doc = Doc(nlp.vocab, words=words, tags=tags)
        except Exception as e:
            print(e)
            print(i, id, tags)
        # for word in doc:
        #     print(word)
        # doc.tags = tags
        example = Example.from_dict(doc, {'words':words, 'tags':tags})

        db.add(doc)
    db.to_disk('./'+data_type+'.spacy')



def test_spacy(ids, poses):
    nlp = spacy.load('../model/spacy/model-best')
    out_list = []
    for id, pos in zip(ids, poses):
        doc = nlp(" ".join(id))
        tags = []
        for word in doc:
            tags.append(word.tag_)
        out_list.append(' '.join(tags))
    return out_list


def create_flair_data(ids, poses, mode):
    with open('./dataset/flair_format/flair_'+mode+'.txt', 'w') as file:
        for id, pos in zip(ids, poses):
            for word, tag in zip(id, pos):
                line = word + ' ' + tag + '\n'
                file.write(line)
            file.write('\n')


def train_flair(train_ids, train_poses, dev_ids, dev_poses, test_ids, test_poses):
    # data
    training_data = []
    create_flair_data(train_ids, train_poses, mode='train')
    create_flair_data(dev_ids, dev_poses, 'dev')
    create_flair_data(test_ids, test_poses, 'test')

    columns = {0: 'text', 1: 'pos'}
    data_folder = './dataset/flair_format/'
    corpus: Corpus = ColumnCorpus(data_folder, columns,
                                  train_file='flair_train.txt',
                                  dev_file='flair_dev.txt',
                                  test_file='flair_test.txt'
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

    trainer.train('./flair_tagger',
                  learning_rate=0.1,
                  mini_batch_size=32)

    sentence = Sentence('get name')
    print(tagger.predict(sentence))
    return tagger

def eva_flair(identifiers, tags):
    tagger = Classifier.load('../model/flair/flair_tagger/best-model.pt')
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
        # print(sentence.get_labels()[0].value)
        # break
        result_tags = [token.value for token in sentence.get_labels()]
        tokens = [token.text for token in sentence.tokens]

        # #wordnet
        # for token in tokens:
        #
        #     w_description = wn.synsets(token)
        #     # print(w_description)
        #     if len(w_description) == 0:
        #         print("fw?"+token)


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
        # if len(result_tags) != len(identifier):
        #     print(result_tags, identifier, flair_input, tag)
        #
        # if tag == result_tags:
        #     correct_ids += 1
        # for j, (golden, out) in enumerate(zip(tag, result_tags)):
        #     if golden == out:
        #         correct_tokens += 1
    print('Flair')
    print("Identifier Accuracy: {}".format(correct_ids/total))
    print("Token Accuracy: {}".format(correct_tokens/total_tokens))
    return out_tags



def preprocess2opennlp(ids, poses, mode):
    out_lines = []
    for i, (id, pos_line) in enumerate(zip(ids, poses)):
        line = []
        for token, pos in zip(id, pos_line):
            pair = token+'_'+pos
            line.append(pair)
        line_str = ' '.join(line) + '\n'
        out_lines.append(line_str)
    with open('./opennlp_'+mode+'.train', 'w') as f:
        f.writelines(out_lines)
    return out_lines


def conllu(ids, poses, name):
    # from nltk.stem import WordNetLemmatizer
    import stanza

    nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma,depparse')
    # wnl = WordNetLemmatizer()
    lines = []
    for i, (form_line, xpos_line) in enumerate(zip(ids, poses)):

        line = ' '.join(form_line)
        doc = nlp(line)
        sent = doc.sentences[0]

        meta_1 = '# sent_id = ' + str(i) + '\n'
        meta_2 = '# text = ' + line + '\n'
        lines.append(meta_1)
        lines.append(meta_2)
        if len(form_line) != len(sent.words):
            print(i)

        # for j, (form, xpos, word) in enumerate(zip(form_line, xpos_line, sent.words)):
        for j in range(len(form_line)):
            if len(sent.words) <= j:
                word = sent.words[-1]
            else:
                word = sent.words[j]
            # id = j

            out_line = (str(j+1)+'\t'+ form_line[j]+'\t')
            # lemma = wnl.lemmatize(form, xpos)
            # lemma = word.lemma
            # print(word.lemma)
            out_line += (word.lemma + '\t')
            # upos = word.upos
            out_line += (word.upos + '\t')
            out_line += word.xpos + '\t'
            feats = word.feats if word.feats else "_"
            out_line += feats+'\t'
            # head = # sent.words[word.head-1].text if word.head > 0 else "root"
            head = word.head if word.head < len(sent.words) else len(sent.words) -1
            out_line += str(head) +'\t'
            deprel = word.deprel
            out_line += deprel +'\t'
            deps = '_'
            out_line += deps + '\t'
            misc = '_'
            out_line += misc + '\n'
            lines.append(out_line)

            # print(f'{j}: {out_line}')
        lines.append('\n')

    with open('./'+'en_test-ud-'+name+'.conllu', 'w') as f:
        f.writelines(lines)


def read_opennlp(tags, mode):
    # mode = config.mode
    # create_opennlp_input(mode, corpus.test_input)
    # opennlp_pos_tag(config, corpus.test_input)
    opennlp_out = parse_opennlp_output(f'./opennlp_output/opennlp_retrain_{mode}_results_new.txt')
    # opennlp_out = parse_opennlp_output(config.opennlp_home+f'/opennlp_{mode}_results_per.txt')

    em, token_acc = evaluate(opennlp_out, tags)
    print('em: ', em)
    print('Token Acc: ', token_acc)
    return list2str(opennlp_out)


if __name__ == '__main__':
    config = Config()
    mode = config.mode
    print(mode)
    data_path = config.data_path
    corpus = Corpus(data_path, mode)

    # nltk
    print('NLTK: ')
    # nltk_tagger = train_nltk(corpus.train_input, corpus.train_tags)
    nltk_out = eva_nltk(corpus.test_input, corpus.test_tags)
    evaluate(str2list(nltk_out), corpus.test_tags)

    # flair
    print("Flair: ")
    # train_flair(corpus.train_input, corpus.train_tags, corpus.dev_input, corpus.dev_tags, corpus.test_input, corpus.test_tags)
    flair_out = eva_flair(corpus.test_input, corpus.test_tags)

    # corenlp
    # create_stanford_data(corpus.train_input, corpus.train_tags)
    # create_stanford_data(corpus.test_input, corpus.test_tags, 'stanford_test'+mode+'.txt')
    # create_stanford_data(corpus.dev_input, corpus.dev_tags, 'stanford_dev.txt')
    print("CoreNLP: ")
    stanford_out = read_stanford('./stanford_output/stanford_'+mode+'_out.txt')
    evaluate(str2list(stanford_out), corpus.test_tags)

    # spacy
    # preprocess2spacy(corpus.train_input, corpus.train_tags)
    # preprocess2spacy(corpus.test_input, corpus.test_tags, mode)
    # preprocess2spacy(corpus.dev_input, corpus.dev_tags, 'dev')
    print('spaCy: ')
    spacy_out = test_spacy(corpus.test_input, corpus.test_tags)
    evaluate(str2list(spacy_out), corpus.test_tags)

    # opennlp
    # preprocess2opennlp(corpus.train_input, corpus.train_tags, 'train')
    # preprocess2opennlp(corpus.test_input, corpus.test_tags, 'test_'+mode)
    # preprocess2opennlp(corpus.dev_input, corpus.dev_tags, 'dev')
    print('OpenNLP: ')
    opennlp_out = read_opennlp(corpus.test_tags, mode)


    # conllu(corpus.train_input, corpus.train_tags, name='train')
    # conllu(corpus.dev_input, corpus.dev_tags, name='dev')
    # conllu(corpus.test_input, corpus.test_tags, name='test')
    data = {'sequence': list2str(corpus.test_input),
            'pos': list2str(corpus.test_tags),
            'nltk': nltk_out,
            'corenlp': stanford_out,
            'opennlp': opennlp_out,
            'spacy': spacy_out,
            'flair': flair_out}
    df = pd.DataFrame(data)
    df.to_csv('./retrain_'+mode+'.csv')

