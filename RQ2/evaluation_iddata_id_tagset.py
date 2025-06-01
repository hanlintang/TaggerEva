import pandas as pd
import sqlite3
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import Config
from corpus import Corpus


def compare(corpus, path='./new_data/et_out'):
    print(os.getcwd())
    et_results = []
    new_et_results = []

    et_results = open('./new_data/et_out/et_out.txt', 'r').readlines()
    for line in et_results:
        new_et_results.append(line.strip().split(' '))

    true_results = []
    for filename in os.listdir('./new_data/method/test'):
        print(filename)
        pos_tags = pd.read_csv(os.path.join('./new_data/method/test', filename))['pos_tag'].values.tolist()[:1000]
        # print(filename, len(et_result))
        true_results += [line.strip().split(' ') for line in pos_tags]
    # true_results = pd.read_csv('./new_data/method/test/pos_nltk_maven_method.csv')['pos_tag'].values.tolist()
    # poses = [line.split(' ') for line in true_results]
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
    print('id: ', corr_id/id_num, corr_id, id_num)


def create_data(train_ids, train_poses):
    # pos_map = pd.read_csv('./map.csv')
    # map_dict = pos_map[['ORI', 'ET']].set_index('ORI').to_dict(orient='dict')['ET']
    columns = ['IDENTIFIER', 'GRAMMAR_PATTERN', 'POSITION', 'NORMALIZED_POSITION',
               'CONTEXT', 'WORD', 'CORRECT_TAG', 'ID']
    # print(map_dict)
    out_lines = []
    id = 0
    for i, (id_list, pos_list) in enumerate(zip(train_ids, train_poses)):
        for j, (token, pos) in enumerate(zip(id_list, pos_list)):
            if pos in ['WP', 'RBR', 'PRP']:
                print(i, id_list, token, pos)
                continue
            line = [' '.join(id_list), ' '.join(pos_list), j]
            # normalized position
            if j == 0:
                line.append(0)
            elif j == len(id_list) - 1:
                line.append(2)
            else:
                line.append(1)
            # context FUNCTION
            line.append(4)
            # word & tag
            line.append(token)
            line.append(pos)
            # id
            line.append(i)
            id += 1
            out_lines.append(line)
    features = list(map(list, zip(*out_lines)))
    features_dict = {}

    for column, data_list in zip(columns, features):
        features_dict[column] = data_list

    et_data = pd.DataFrame(data=features_dict)
    et_data.to_csv('./train_ori_to_et.csv')
    # print(et_data)
    conn = sqlite3.connect('./train_ori_to_et.db')
    et_data.to_sql('base', conn, if_exists='replace', index=False)
    return et_data


def create_et_data(config, train_ids, train_poses):
    pos_map = pd.read_csv('./map.csv')
    map_dict = pos_map[['ORI', 'ET']].set_index('ORI').to_dict(orient='dict')['ET']
    # columns = ['IDENTIFIER', 'GRAMMAR_PATTERN', 'POSITION', 'NORMALIZED_POSITION',
    #            'CONTEXT', 'WORD', 'CORRECT_TAG', 'ID']
    columns = ['IDENTIFIER', 'GRAMMAR_PATTERN', 'POSITION', 'NORMALIZED_POSITION',
               'CONTEXT', 'WORD', 'CORRECT_TAG', 'ID', 'TYPE']
    context2num = {'class':2, 'method':4, 'args':5}
    context = context2num[config.mode]
    print(map_dict)
    out_lines = []
    id = 0
    for i, (id_list, pos_list) in enumerate(zip(train_ids, train_poses)):
        map_pos_list = []

        for pos in pos_list:
            if pos not in map_dict.keys():
                map_pos_list.append('UNK')
            else:
                map_pos_list.append(map_dict[pos])
        for j, (token, pos) in enumerate(zip(id_list, map_pos_list)):
            if pos == 'UNK':
                continue
            line = [' '.join(id_list), ' '.join(map_pos_list), j]
            # normalized position
            if j == 0:
                line.append(0)
            elif j == len(id_list) - 1:
                line.append(2)
            else:
                line.append(1)
            # context FUNCTION 4, DECLLARATION 3, CLASS 2, PARAMETER 5
            line.append(context)
            # word & tag
            line.append(token)
            line.append(pos)
            # id
            line.append(i)
            id += 1
            # type

            out_lines.append(line)
    features = list(map(list, zip(*out_lines)))
    features_dict = {}

    for column, data_list in zip(columns, features):
        features_dict[column] = data_list

    et_data = pd.DataFrame(data=features_dict).sample(frac=1)
    et_data.to_csv(f'./{config.mode}_to_et.csv')
    # print(et_data)
    conn = sqlite3.connect(f'./{config.mode}_to_et.db')
    et_data.to_sql('base', conn, if_exists='replace', index=False)
    return et_data


def compare_st(path='./new_data/et_out/'):
    ori_st_path = os.path.join(path, 'add_et.csv')
    ret_st_path = os.path.join(path, 'add_et_retrain.csv')

    ret_st = pd.read_csv(ret_st_path)
    true_pos_list = ret_st['pos'].values.tolist()
    st_out_list = ret_st['et'].values.tolist()

    count_wrong_JJ = 0
    count_NN = 0
    for i, (pos_line, st_line) in enumerate(zip(true_pos_list, st_out_list)):
        poses = pos_line.split(' ')
        line = st_line.split(' ')
        for j, (pos, out) in enumerate((zip(poses, line))):
            if pos == 'NN':
                count_NN += 1
                if out == 'JJ':
                    count_wrong_JJ += 1

    print(count_wrong_JJ, count_NN, count_wrong_JJ/count_NN)


def swum_posse_results(path='./out_data_et.csv', golden_path='./evaluation.csv', tagger='SWUM', with_unk=True):
    w_num = 0
    df = pd.read_csv(path)
    data = pd.read_csv(golden_path)
    pos_map = pd.read_csv('./map.csv')
    map_dict = pos_map[['ORI', 'ET']].set_index('ORI').to_dict(orient='dict')['ET']

    outs = df[tagger].values.tolist()
    # outs = data['flair'].values.tolist()
    outs = [line.split(' ') for line in outs]
    # print(outs)
    poses = data['POS'].values.tolist()
    poses = [line.split(' ') for line in poses]
    # print(poses)
    corr_token = 0
    corr_id = 0
    id_num = len(outs)
    token_num = 0
    for i, (line, et) in enumerate(zip(poses, outs)):
        new_et = []
        id_len = len(line)
        pos_to_et_line = []
        # temp_et = []
        unk_flag = False
        for w in line:
            if w not in map_dict.keys():
                pos_to_et_line.append('UNK')
                if with_unk == False:
                    unk_flag = True
                    break
            else:
                pos_to_et_line.append(map_dict[w])

        if unk_flag == True and with_unk == False:
            id_num -= 1
            continue

        # for w in et:
        #     if w not in map_dict.keys():
        #         temp_et.append('UNK')
        #     else:
        #         temp_et.append(map_dict[w])
        temp_et = et

        # print(temp_et)

        if len(pos_to_et_line) != len(temp_et):
            # print('length error', i, line, temp_et)
            for j in range(id_len):
                if j >= len(temp_et):
                    new_et.append('PAD')
                    continue
                new_et.append(temp_et[j])
        else:
            new_et = temp_et

        # print(pos_to_et_line, new_et)
        assert len(pos_to_et_line) == len(new_et)
        if ' '.join(pos_to_et_line) == ' '.join(new_et):
            corr_id += 1
        for token, e_token in zip(pos_to_et_line, new_et):
            token_num += 1
            if token == e_token:
                corr_token += 1
            else:
                w_num += 1
                # print(i, w_num, token, e_token, new_et, et)
    print(tagger)
    print('token: ', corr_token/token_num, corr_token, token_num)
    print('id: ', corr_id/id_num, corr_id, id_num)


def create_posse_data():
    et_data = pd.read_csv('./test_input.csv')
    types = et_data['TYPE'].values.tolist()
    declarations = et_data['DECLARATION'].values.tolist()
    data = pd.read_csv('./evaluation.csv')
    ids = data['sentence'].values.tolist()
    lines = []
    for i, id in enumerate(ids):
        line = types[i].strip() + ' ' + declarations[i].strip() + ' | ' + id + '\n'
        lines.append(line)
    print(lines)
    with open('./posse_data.input', 'w') as f:
        f.writelines(lines)


def cal_posse():
    true_data = pd.read_csv('./evaluation.csv')
    ids = true_data['sentence'].values.tolist()
    poses = true_data['pos'].values.tolist()
    pos_lines = []
    new_lines = []
    with open('./posse_data.input.pos', 'r') as f:
        lines = f.readlines()
        for line in lines:
            temp_line = line.split('|')[-1].strip().split(',')
            pos_line = [w.split(':')[-1] for w in temp_line]
            pos_lines.append(pos_line)
            new_lines.append(temp_line)
        print(len(pos_lines))
        print(pos_lines[10])
    pos_tags = set([p for line in pos_lines for p in line])
    print(pos_tags, len(pos_tags))

    pos_map = pd.read_csv('./posse_map.csv')
    map_dict = pos_map[['ORI', 'POSSE']].set_index('ORI').to_dict(orient='dict')['POSSE']

    outs = pos_lines
    print(outs)

    poses = [line.split(' ') for line in poses]
    print(poses)
    corr_token = 0
    corr_id = 0
    id_num = len(outs)
    token_num = 0
    for i, (line, et) in enumerate(zip(poses, outs)):
        new_et = []
        id_len = len(line)
        pos_to_et_line = []
        for w in line:
            if w not in map_dict.keys():
                pos_to_et_line.append('UNK')
            else:
                pos_to_et_line.append(map_dict[w])
        if len(pos_to_et_line) != len(et):
            # print('length error', i, line, et)
            for j in range(id_len):
                if j >= len(et):
                    new_et.append('PAD')
                    continue
                new_et.append(et[j])
        else:
            new_et = et

        # print(pos_to_et_line, new_et)
        assert len(pos_to_et_line) == len(new_et)
        if ' '.join(pos_to_et_line) == ' '.join(new_et):
            corr_id += 1
        for token, e_token in zip(pos_to_et_line, new_et):
            token_num += 1
            if token == e_token:
                corr_token += 1
            #else:
            #     w_num += 1
            #     print(i, token, e_token, pos_to_et_line, new_et, ids[i])
        #
    print('POSSE')
    print('token: ', corr_token/token_num, corr_token, token_num)
    print('id: ', corr_id/id_num, corr_id, id_num)


def nl_tagger2et(tagger, mode, with_unk=True):
    w_num = 0
    path = '../evaluation_results/evaluation_'+mode+'.csv'
    df = pd.read_csv(path)
    data = df
    pos_map = pd.read_csv('./map.csv')
    map_dict = pos_map[['ORI', 'ET']].set_index('ORI').to_dict(orient='dict')['ET']

    outs = df[tagger].values.tolist()
    outs = [line.split(' ') for line in outs]
    poses = data['POS'].values.tolist()
    poses = [line.split(' ') for line in poses]
    corr_token = 0
    corr_id = 0
    id_num = len(outs)
    token_num = 0
    for i, (line, et) in enumerate(zip(poses, outs)):
        new_et = []
        id_len = len(line)
        pos_to_et_line = []
        temp_et = []
        unk_flag = False
        for w, tag in zip(line, et):
            if w not in map_dict.keys():
                # pos_to_et_line.append('UNK')
                if with_unk == False:
                    unk_flag = True
                    break
                pos_to_et_line.append(w)
            else:
                pos_to_et_line.append(map_dict[w])
            if tag not in map_dict.keys():
                # temp_et.append('UNK')

                temp_et.append(tag)
            else:
                # try:
                temp_et.append(map_dict[tag])
                # except Exception as e:
                #     print(tag not in map_dict.keys())
                #     print(tag)
                #     print(map_dict.keys())

        # print(temp_et)
        if unk_flag == True and with_unk == False:
            id_num -=1
            continue

        if len(pos_to_et_line) != len(temp_et):
            # print('length error', i, line, temp_et)
            for j in range(id_len):
                if j >= len(temp_et):
                    new_et.append('PAD')
                    continue
                new_et.append(temp_et[j])
        else:
            new_et = temp_et

        # print(pos_to_et_line, new_et)
        assert len(pos_to_et_line) == len(new_et)
        if ' '.join(pos_to_et_line) == ' '.join(new_et):
            corr_id += 1
        for token, e_token in zip(pos_to_et_line, new_et):
            token_num += 1
            if token == e_token:
                corr_token += 1
            else:
                w_num += 1
                # print(i, w_num, token, e_token, new_et, et)
    print(tagger)
    print('token: ', corr_token/token_num, corr_token, token_num)
    print('id: ', corr_id/id_num, corr_id, id_num)



if __name__ == '__main__':
    config = Config()
    # nl2et
    out_files = ['method', 'args', 'class', 'all']

    taggers = ['NLTK', 'CORENLP', 'OPENNLP', 'SPACY', 'FLAIR', 'STANZA']
    for tagger in taggers:
        print(tagger)
        for mode in out_files:
            print(mode)
            nl_tagger2et(tagger, mode, with_unk=config.with_unk)
