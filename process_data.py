import spacy
from spacy.tokens import Doc, DocBin
from spacy.training import Example

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