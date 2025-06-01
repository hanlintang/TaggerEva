import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from collections import Counter

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import evaluate, str2list
from config import Config

def drop_fw(path='../evaluation_results/evaluation_all.csv'):
    print(path)
    df = pd.read_csv(path)
    filtered_df = df[~df['POS'].str.contains('FW')]
    true_poses = str2list(filtered_df['POS'].values.tolist())

    taggers = ['NLTK', 'CORENLP', 'OPENNLP', 'SPACY', 'FLAIR', 'STANZA']
    em_list = []
    acc_list = []
    for tagger in taggers:
        out_poses = str2list(filtered_df[tagger].values.tolist())
        em, acc = evaluate(out_poses, true_poses)
        em_list.append(em)
        acc_list.append(acc)
    print(taggers)
    print('EM: ', em_list)
    print('Token Acc: ', acc_list)


def pos_token_stats(csv_path, pos_tag):

    df = pd.read_csv(csv_path)


    total_tokens = 0
    total_items = len(df)
    pos_token_count = 0
    pos_item_count = 0


    for index, row in df.iterrows():
        # tokens = row['SEQUENCE'].split()
        # pos_tags = row['POS'].split()
        pos_tags = row['POSSE'].split()

        # total_tokens += len(tokens)


        current_pos_count = pos_tags.count(pos_tag)
        pos_token_count += current_pos_count


        if current_pos_count > 0:
            pos_item_count += 1


    # token_ratio = pos_token_count / total_tokens if total_tokens > 0 else 0
    item_ratio = pos_item_count / total_items if total_items > 0 else 0


    print(f"POS '{pos_tag}' in all items：{item_ratio * 100:.2f}%, nums：{pos_item_count}")

    # return token_ratio, item_ratio


def extract_pos_items(csv_path, pos_tag, output_csv):
    df = pd.read_csv(csv_path)

    extracted_items = []


    for index, row in df.iterrows():
        tokens = row['SEQUENCE'].split()
        pos_tags = row['POS'].split()


        matched_tokens = [token for token, pos in zip(tokens, pos_tags) if pos == pos_tag]


        if matched_tokens:
            extracted_items.append({
                'SEQUENCE': row['SEQUENCE'],
                'POS': row['POS'],
                'Matched Tokens': ' '.join(matched_tokens)
            })


    extracted_df = pd.DataFrame(extracted_items)
    extracted_df.to_csv(output_csv, index=False)

    print(f"save to: {output_csv}")


def calculate_fw_proportion(csv_path):
    df = pd.read_csv(csv_path)

    fw_count = len(df[df['POS']=='FW'])

    total_count = len(df)

    fw_proportion = fw_count / total_count if total_count > 0 else 0

    print(f"items including 'FW' : {fw_count}")
    print(f"'FW' items proportion: {fw_proportion * 100:.2f}%")

    return fw_count, fw_proportion

def plot_category_distribution(csv_path):
    matplotlib.use('Agg')
    # readcsv
    df = pd.read_csv(csv_path)

    # read_last 4 columns
    last_4_columns = df.iloc[:, -4:]

    # flatten and remove NaN
    values = last_4_columns.values.flatten()
    values = [v for v in values if pd.notna(v)]

    # count for labels
    category_counts = Counter(values)

    labels = ['Software Development' , 'Standard', 'Software/Library', 'System/Hardware', 'Operator', 'Other']

    plt.figure(figsize=(8, 8))
    explode = tuple([0.05] * len(labels))
    color_name = 'Set3'
    colors = plt.get_cmap(color_name)
    color_list = [colors((i) % len(labels)) for i in range(len(labels))]
    plt.pie(category_counts.values(), labels=[labels[int(key)] for key in category_counts.keys()], autopct='%1.2f%%', explode=explode, colors=color_list)
    # ax = plt.pie(category_counts.values(), labels=labels, autopct='%1.2f%%', explode=explode)

    # colormap to every bar
    # pies = ax.patches
    # for i, bar in enumerate(ax):
    #     bar.set_color(colors((i) % len(ax)))

    plt.legend(loc=1, bbox_to_anchor=(1.12,1.1),borderaxespad = 0.)
    # plt.legend(loc=1)
    # plt.title('Category Distribution in Last Four Columns')
    # plt.axis('equal')
    plt.savefig('./nnp.pdf', format='pdf')
    plt.show()

if __name__ == '__main__':
    config = Config()
    mode = config.mode
    print(f'Mode: {mode}')
    # drop_fw('./evaluation_method.csv')
    # drop_fw('./evaluation_args.csv')
    # drop_fw('./evaluation_class.csv')
    drop_fw(f'../evaluation_results/evaluation_{config.mode}.csv')
    #
    # calculate_fw_proportion('./evaluation_args.csv')
    #
    # pos_token_stats('./evaluation_all.csv', 'FW')
    # pos_token_stats('./evaluation_method.csv', 'FW')
    # pos_token_stats('./evaluation_args.csv', 'FW')
    # pos_token_stats('./evaluation_class.csv', 'FW')
    #
    # pos_token_stats('./evaluation_all.csv', 'NNP')
    # pos_token_stats('./evaluation_all.csv', 'NNPS')
    # pos_token_stats('./dataset/et_out_data/et_out_ALL_new.csv', 'FAILURE')

    # extract_pos_items('./evaluation_all.csv', 'NNP', 'nnp_all.csv')
    # extract_pos_items('./evaluation_all.csv', 'NNPS', 'nnps_all.csv')

    # plot_category_distribution('./nnp_nnps.csv')
