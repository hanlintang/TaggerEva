# TaggerEva
========

## Introduction
* Experimental Data:
    - IDData: The test dataset introduced in this study.
    - MNTrain: The training dataset used for training POS taggers.
    - NLData: The natural language dataset used in selected experiments.
    - NNP/NNPS Classification Results: Classification results for proper nouns and plural proper nouns.

* Source Code:
    - Complete experimental code and scripts organized by RQs, with a dedicated folder for each RQ.
    - Installation instructions and Docker image configurations to facilitate reproducibility.

* Experimental Results:
    - Input data formatted for OpenNLP, CoreNLP, spaCy, Flair, and the Ensemble Tagger, provided in separate directories.
    - Output results of all POS taggers stored in the ```evaluation_results``` directory for reference and further analysis.
    - Model parameter files trained on the MNTrain dataset, available under the ```model``` directory.

## Install Selected Taggers
### NL Taggers
* [NLTK](https://www.nltk.org/install.html)
* [Stanford CoreNLP](https://stanfordnlp.github.io/CoreNLP/)
* [Apache OpenNLP](https://opennlp.apache.org/)
* [spaCy](https://spacy.io/)
* [Flair](https://github.com/flairNLP/flair)
* [Stanza](https://stanfordnlp.github.io/stanza/)
  
### ID Taggers
* [SWUM](https://github.com/SCANL/SWUM)
* [POSSE](https://github.com/SCANL/POSSE) 
* [Ensemble](https://github.com/SCANL/ensemble_tagger)

## Dataset
Please click [data](./dataset/README.md) to read the introduction of the TaggerEva dataset. 

## Setup
### Docker (Recommended)
#### Ubuntu
1. [Click here to download the image.](https://drive.google.com/file/d/1cDxHUTV357YbMwWlXO4EXhrfCzAdCWHM/view?usp=sharing)

2. Configure the docker image and run it:
```shell
gunzip -c taggereva_image.tar.gz | docker load
docker run -it --name taggereva taggereva-image:1.0 /bin/bash
```

3. All programs/scripts are located in ```/home/taggereva```
```shell
cd /home/taggereva
```
Follow the instructions of each RQ.

### Manual installation
1. Install the dependencies:
```sudo pip3 install -r requirements.txt```
   
2. Download the model of spaCy:
```python -m spacy download en_core_web_sm```

3. Install Spiral following [Spiral Repo](https://github.com/casics/spiral).

4. Change the ```data_path``` in ```config.py``` into your data path.

## RQs
Please follow the instruction of each RQ.

* [RQ1](./RQ1/README.md) 
* [RQ2](./RQ2/README.md) 
* RQ3
* [RQ4](./RQ4/README.md) 
* [RQ5/6](./RQ56/README.md) 

## Evaluation of Taggers
### Natural language Taggers
Need to run OpenNLP first.

#### OpenNLP
OpenNLP needs command line:
1. Download the default maxent [model](https://opennlp.sourceforge.net/models-1.5/en-pos-maxent.bin) into its installation path.
2. Run the following command and copy the output file into the project.
```
./opennlp POSTagger ../models/en-pos-maxent.bin < ../opennlp_format/opennlp_{id-type/nl}_input.txt > opennlp_{id-type/nl}_results.txt
```


## Train Taggers

### CoreNLP & OpenNLP & spaCy
These two taggers need to be trained by the command line interface.
#### Stanford CoreNLP
* Copy the stanford_format data and "./model/stanford/maxnet.props" into the installation directory of Stanford CoreNLP.
* Run the command in command line:
```
java -mx1g -cp "*" edu.stanford.nlp.tagger.maxent.MaxentTagger -prop maxent.props -model "retrain_stanford.model" -testFile "stanford_test.txt" > stanford_out.txt
```
* Copy the "stanford_out.txt" back to TaggerEva

#### OpenNLP
* Copy the opennlp_format data into the installation path of opennlp.
* Run the command in command line:
```
./opennlp POSTaggerTrainer -model en-pos-maxent-retrain.bin -lang en -data ./opennlp_format/MNTrain.train -encoding UTF-8
./opennlp POSTagger ../models/en-pos-maxent-retrain.bin < ../opennlp_format/opennlp_{id-type}_input.txt > opennlp_retrain_{id-type}_results.txt
```
* Copy the output file back and run train.py to parse it.

#### spaCy
 
  ```
  cd ./model/spacy/
  python -m spacy train spacy_config.cfg --paths.train ../dataset/spacy_format/train.spacy --paths.dev ../dataset/spacy_format/dev.spacy --output ./
```

After training, you can run the command:
```
python train.py -m method/args/class/all
```
for evaluation.

## Model
The nltk, corenlp, opennlp and spacy retrained model has stored in "model". Due to the size limitation of Github, the flair model currently not been committed.
