import pandas as pd
import subprocess

import os
# import sys
# sys.path.append("/home/ensemble_tagger/ensemble_tagger_implementation")

from ensemble_functions import Annotate_word, Run_external_taggers
from process_features import Calculate_normalized_length, Add_code_context


CONTEXT = ['ATTRIBUTE', 'CLASS', 'DECLARATION', 'FUNCTION', 'PARAMETER']


def et_tag(identifier_type, identifier_name, identifier_context):
    output = []
    swum = []
    posse = []
    stanford = []
    try:
        ensemble_input = Run_external_taggers(identifier_type + ' ' + identifier_name, identifier_context)
        print(ensemble_input)
        for key, value in ensemble_input.items():
            swum.append(value[0])
            posse.append(value[1])
            stanford.append(value[2])

        ensemble_input = Calculate_normalized_length(ensemble_input)
        ensemble_input = Add_code_context(ensemble_input, identifier_context)
        for key, value in ensemble_input.items():
            result = Annotate_word(value[0], value[1], value[2], value[3], value[4].value)
            # output.append("{identifier},{word},{swum},{posse},{stanford},{prediction}"
            # .format(identifier=(identifier_name),word=(key),swum=value[0], posse=value[1], stanford=value[2], prediction=result))
            # output.append("{word}|{prediction}".format(word=(key[:-1]),prediction=result))
            output.append(result)
    except Exception as e:
        print(identifier_name, 'failure')
        output.append('UNK')
        swum.append('UNK')
        posse.append('UNK')
        stanford.append('UNK')

    return output, swum, posse, stanford


def evaluate_et(et_data):
    identifiers = et_data['IDENTIFIER'].values.tolist()
    contexts = et_data['CONTEXT'].values.tolist()
    tags = et_data['GRAMMAR_PATTERN'].values.tolist()

    et_outs, swum_outs, posse_outs, stanford_outs = [], [], [], []
    for i in range(len(identifiers)):
        identifier_name = ''.join(identifiers[i].split(' '))
        print(identifier_name)
        identifier_context = CONTEXT[contexts[i] - 1]
        if identifier_context == 'CLASS':
            identifier_type = 'class'
        else:
            identifier_type = 'int'
        tag = tags[i]
        et, swum, posse, stanford = et_tag(identifier_type, identifier_name, identifier_context)

        et_str = ' '.join(et)
        et_outs.append(et_str)
        swum_str = ' '.join(swum)
        swum_outs.append(swum_str)
        posse_str = ' '.join(posse)
        posse_outs.append(posse_str)
        stanford_str = ' '.join(stanford)
        stanford_outs.append(stanford_str)

        if len(et_outs) != len(swum_outs):
            print(f'{i}, {identifier_name}, {len(et_outs), len(swum_outs), len(posse_outs), len(stanford_outs)}')
            print(et_outs)
            print(swum_outs)
            # print(i, len(outputs))

    data = {'ENSEMBLE_TAGGER': et_outs, 'SWUM': swum_outs, 'POSSE': posse_outs, 'STANFORD': stanford_outs, 'POS': tags}
    print(len(et_outs), len(swum_outs), len(posse_outs), len(stanford_outs))

    df = pd.DataFrame(data)

    df.to_csv(f'et_data_out.csv')


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


if __name__=='__main__':
    command = 'export PERL5LIB=/home/ensemble_tagger/POSSE/Scripts'
    result = subprocess.run(command, shell=True)
    print(result.returncode)

    et_data = pd.read_csv('../dataset/ensemble_tagger_training_data/unseen_testing_data.csv')
    et_data = et_data.drop_duplicates(subset='IDENTIFIER', keep='first')
    evaluate_et(et_data)

    df = pd.read_csv('./et_data_out.csv')
    et_outs = df['ENSEMBLE_TAGGER'].values.tolist()
    swum_outs = df['SWUM'].values.tolist()
    posse_outs = df['POSSE'].values.tolist()
    stanford_outs = df['STANFORD'].values.tolist()
    poses = df['POS'].values.tolist()

    outs = [et_outs, swum_outs, posse_outs, stanford_outs]
    taggers = ['ensemble tagger', 'swum', 'posse', 'stanford']

    for tagger, out in zip(taggers, outs):
        print(tagger)
        cal_acc_em(out, poses)
