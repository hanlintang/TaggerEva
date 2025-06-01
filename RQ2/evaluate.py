import pandas as pd
import subprocess

import os
# import sys
# sys.path.append("/home/ensemble_tagger/ensemble_tagger_implementation")

from ensemble_functions import Annotate_word, Run_external_taggers
from process_features import Calculate_normalized_length, Add_code_context


if __name__=='__main__':
    # command = 'export PERL5LIB=/home/ensemble_tagger/POSSE/Scripts'
    # result = subprocess.run(command, shell=True)
    # print(result.returncode)

    for file in os.listdir('/home/taggereva/dataset/ensemble_format'):
        outputs = []
        df = pd.read_csv('/home/taggereva/dataset/ensemble_format/'+file)
        print(file)
        identifier_context = 'FUNCTION'
        if 'method' in file:
            identifier_context = 'FUNCTION'
        elif 'args' in file:
            identifier_context = 'PARAMETER'
        elif 'class' in file:
            identifier_context = 'CLASS'

        assert identifier_context
        print(identifier_context)
        length = len(df)
        types = df['TYPE'].values.tolist()
        names = df['DECLARATION'].values.tolist()
        posse_outs = []
        swum_outs = []
        stanford_outs = []
        for i, (identifier_type, identifier_name) in enumerate(zip(types, names)):
            posse = []
            swum = []
            stanford = []

            # if (i+1)%50 ==0:
            #     print(file, i)
            # identifier_type = df.iloc[i]['return'].strip()
            # identifier_name = df.iloc[i]['declaration'].strip()
            if type(identifier_type) is not str:
                print('error type: ', identifier_type)
                identifier_type = 'void'
            identifier_name = identifier_name.strip()
            identifier_type = identifier_type.strip()
            if 'throws' in identifier_name:
                identifier_name = identifier_name[:identifier_name.find('throws')].strip()
                # print(identifier_type, identifier_name)
            # print(identifier_type, identifier_name)
            output = []
            try:
                ensemble_input = Run_external_taggers(identifier_type + ' ' + identifier_name, identifier_context)
                print(ensemble_input)
                for key, value in ensemble_input.items():
                    swum.append(value[0])
                    posse.append(value[1])
                    stanford.append(value[2])

                ensemble_input = Calculate_normalized_length(ensemble_input)
                ensemble_input = Add_code_context(ensemble_input,identifier_context)
                for key, value in ensemble_input.items():
                    result = Annotate_word(value[0], value[1], value[2], value[3], value[4].value)
                #output.append("{identifier},{word},{swum},{posse},{stanford},{prediction}"
                #.format(identifier=(identifier_name),word=(key),swum=value[0], posse=value[1], stanford=value[2], prediction=result))
                # output.append("{word}|{prediction}".format(word=(key[:-1]),prediction=result))
                    output.append(result)
            except Exception as e:
                print(identifier_name, 'failure')
                output.append('UNK')
                swum.append('UNK')
                posse.append('UNK')
                stanford.append('UNK')
                # swum_outs.append(['UNK'])
                # posse_outs.append(['UNK'])
                # stanford_outs.append(['UNK'])

            # output_str = ' '.join(output)+'\n'
            output_str = ' '.join(output)
            # print(i, output_str)
            outputs.append(output_str)
            swum_str = ' '.join(swum)
            swum_outs.append(swum_str)
            posse_str = ' '.join(posse)
            posse_outs.append(posse_str)
            stanford_str = ' '.join(stanford)
            stanford_outs.append(stanford_str)

            if  len(outputs) != len(swum_outs):
                print(f'{i}, {identifier_name}, {len(outputs), len(swum_outs), len(posse_outs), len(stanford_outs)}')
                print(outputs)
                print(swum_outs)
                         # print(i, len(outputs))


        data = {'OUT':outputs, 'SWUM':swum_outs, 'POSSE':posse_outs, 'STANFORD':stanford_outs}
        print(len(outputs), len(swum_outs), len(posse_outs), len(stanford_outs))

        df = pd.DataFrame(data)

        df.to_csv(f'et_out_{identifier_context}_new.csv')

        # with open(file + '.txt', 'w') as f:
        #     f.writelines(outputs)

        # with open('swum.txt', 'w') as f:
        #     f.writelines(outputs)
        # with open('posse.txt', 'w') as f:
        #     f.writelines(outputs)
        # with open(file + '.txt', 'w') as f:
        #     f.writelines(outputs)
