# RQ2
Comparison of the NL taggers and ID taggers.

## NL taggers on IDData with ID tagset
1. Edit the ```with_unk=True/False``` to evaluate with/without items including "UNK".
2. Run the script to get the results.
```shell
python evaluation_iddata_id_tagset.py
```

## ID taggers on IDData with ID tagset
1. Activate the python3.8 venv of Ensemble Tagger
```shell
source /home/ensemble_tagger/ensemble_tagger_implementation/ensemble_env/bin/activate
```

2. Link POSSE to PERL5LIB
```shell
export PERL5LIB=/home/ensemble_tagger/POSSE/Scripts
```

3. Copy evaluate.py to ```/home/ensemble_tagger/ensemble_tagger_implementation/```

4. Run the script and get the result csv files
```shell
python evaluate.py
```

5. Run the following command
```shell
deactivate
```

5. Copy these csv files to ```ensemble_output_IDData```

6. Run the evaluation script **(you can directly run this script with current data)**
```shell
python calculate_id_tagger_IDData.py
```

## NL taggers retrained on etdata
1. You need to copy the ```ensemble_tagger_training_data``` into ```dataset/``` from the [repo](https://github.com/SCANL/datasets).

2. Get the evaluation results from taggers retrained with ETData:
```shell
python train_et_data.py
```

3. (Optional) Retrained with ETData:
    *  NLTK/Flair: Remove the comment at the specified position.
    ```python
    # NLTK
    # train_nltk(training_path)
    ...
    # flair
    # train_flair(train_data, test_data)
    ```
    * CoreNLP/spaCy: Follow the "Train taggers" section of [Project Instrction](../README.md).
