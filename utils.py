import os
from itertools import chain
import pandas as pd
import spacy
from spacy.tokens import Doc, DocBin
from spacy.training import Example


def list2str(lists):
    new_lists = []
    for l in lists:
        new_lists.append(' '.join(l))
    return new_lists

def str2list(str_list):
    return [line.split(' ') for line in str_list]

def cal_metrics(path='./retrain_out.csv'):
    # taggers = ['nltk', 'stanford', 'spacy', 'flair']
    df = pd.read_csv(path)
    true_poses = df['pos'].values.tolist()
    tagger_outs = []
    new_outs = []
    for tagger in ['nltk', 'stanford', 'spacy', 'flair']:
        tagger_outs.append(df[tagger].values.tolist())

    em_num = [0, 0, 0, 0]
    token_acc = [0, 0, 0, 0]
    id_num = len(true_poses)
    token_num = len([pos for pos_line in true_poses for pos in pos_line.split(' ')])

    for i, true_pos in enumerate(true_poses):
        true_pos_tokens = true_pos.split(' ')
        taggers_out = [tagger[i].split(' ') for tagger in tagger_outs]
        for tagger, tagger_out in enumerate(taggers_out):
            if true_pos == ' '.join(tagger_out):
                em_num[tagger] += 1
        for j, token_pos in enumerate(true_pos_tokens):
            outs = [out[j] for out in taggers_out]
            for tagger, tagger_out in enumerate(outs):
                if token_pos == tagger_out:
                    token_acc[tagger] += 1

    print('EM: ', em_num[0]/id_num, em_num[1]/id_num, em_num[2]/id_num, em_num[3]/id_num)
    print('Token Acc: ', token_acc[0]/token_num, token_acc[1]/token_num, token_acc[2]/token_num, token_acc[3]/token_num)


def compare(corpus, path='./et_out.txt'):
    print(os.getcwd())
    et_results = []
    new_et_results = []


    et_results = open('./et_out.txt', 'r').readlines()
    for line in et_results:
        new_et_results.append(line.strip().split(' '))

    true_results = []


    pos_tags = pd.read_csv('./dataset/mdata.csv')['POS'].values.tolist()
        # print(filename, len(et_result))
    true_results += [line.strip().split(' ') for line in pos_tags]

    poses = true_results

    print(new_et_results)
    pos_map = pd.read_csv('./map.csv')
    map_dict = pos_map[['ORI', 'ET']].set_index('ORI').to_dict(orient='dict')['ET']
    corr_token = 0
    corr_id = 0
    id_num = len(new_et_results)
    token_num = 0
    for i, (line, et) in enumerate(zip(poses, new_et_results)):

        new_et = []
        for w in line:
            if w not in map_dict.keys():
                new_et.append('UNK')
            else:
                new_et.append(map_dict[w])
        if len(line) != len(et):
            print('length error', i, line, et)
            len_del = len(line) - len(et)
            if len_del > 0:
                for _ in range(len_del):
                    new_et.pop()
            elif len_del < 0:
                for _ in range(-len_del):
                    new_et.append('UNK')


        assert len(new_et) == len(et)
        if ' '.join(new_et) == ' '.join(et):
            corr_id += 1
        for token, e_token in zip(new_et, et):
            token_num += 1
            if token == e_token:
                corr_token += 1
    print('token: ', corr_token/token_num, corr_token, token_num)
    print('EM: ', corr_id/id_num, corr_id, id_num)


def evaluate(out_tags, true_tags):
    # if type(out_tags[0]) is str:
    #     new_tags = [line_tags.strip().split(' ') for line_tags in out_tags]
    #     out_tags = new_tags
    #     print('abc')
    #
    # if type(true_tags[0]) is str:
    #     new_tags = [line_tags.strip().split(' ') for line_tags in true_tags]
    #     true_tags = new_tags
    #     print('abc')
    token_num = len(list(chain(*true_tags)))
    id_num = len(true_tags)
    true_token = 0
    true_id = 0

    for i, (out_line, true_line) in enumerate(zip(out_tags, true_tags)):
        # EM
        if ' '.join(out_line) == ' '.join(true_line):
            true_id += 1
        # token acc
        if len(out_line) > len(true_line):
            out_line = out_line[:len(true_line)]
        elif len(out_line) < len(true_line):
            out_line += ['UNK'] * (len(true_line)-len(out_line))
        for j, (out_tag, true_tag) in enumerate(zip(out_line, true_line)):
            if out_tag == true_tag:
                true_token += 1

    em = true_id / id_num
    token_acc = true_token /token_num
    print(f'em: {em}')
    print(f'token_acc: {token_acc}')
    return em, token_acc



if __name__ == '__main__':
    cal_metrics()
    compare()