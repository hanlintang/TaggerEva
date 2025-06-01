from nltk.tag import pos_tag
from itertools import chain
from stanfordcorenlp import StanfordCoreNLP
import spacy
from flair.nn import Classifier
from flair.models import SequenceTagger
from flair.data import Sentence
import stanza


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
    # tagger = Classifier.load('/home/hltang/.flair/models/pos-english/pytorch_model.bin')
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