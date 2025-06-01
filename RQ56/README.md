# RQ5/6

1. Evaluation of taggers retrained by MNTrain:
```shell
python train.py
```

2. (Optional) Retrain with MNTrain:
    *  NLTK/Flair: Remove the comment at the specified position.
    ```python
    # nltk
    # nltk_tagger = train_nltk(corpus.train_input, corpus.train_tags)
    ...
    # flair
    # train_flair(corpus.train_input, corpus.train_tags, corpus.dev_input, corpus.dev_tags, corpus.test_input, corpus.test_tags)
    ```
    * CoreNLP/OpenNLP/spaCy: Follow the "Train taggers" section of [Project Instrction](../README.md).
