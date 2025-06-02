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
#### Create a container from the Docker image
The Docker image can be obtained in two ways:
##### Pull from cloud container registry
1. Run the pull command:
```shell
docker pull registry.cn-beijing.aliyuncs.com/taggereva/taggereva-image
```

2. Create a container:
```shell
docker run -it --name taggereva registry.cn-beijing.aliyuncs.com/taggereva/taggereva-image
```

##### Download image file
1. [Click here to download the image.](https://drive.google.com/file/d/1y748XTYFa1hMLVBOCD0-q050Owwx9Z6X/view?usp=sharing)


2. Configure the docker image and run it:

* Ubuntu: 
```shell
gunzip -c taggereva_image_v1.2.tar.gz | docker load
docker run -it --name taggereva taggereva_image:1.2 /bin/bash
```

* WSL
    1. Install [7-zip](https://www.7-zip.org/) to unpack ```.tar.gz```;
    2. Open Docker Desktop;
    3. Load the image from tar file on PowerShell:
    ```shell
    docker load -i .\taggereva_image_v1.2.tar
    ```
    4. Create a container:
    ```shell
    docker run -it --name taggereva taggereva_image:v1.2
    ```

* macOS
    1. Open Docker Desktop;
    2. Same with Ubuntu.

#### Execute
1. If the container already exists:
```shell
docker start taggereva
```

```shell
docker exec -it taggereva /bin/bash
```

2. All programs/scripts are located in ```/home/taggereva```
```shell
cd /home/taggereva
```
Follow the instructions of each RQ.

Screen records:

[Windows](https://drive.google.com/file/d/1iuG73_zHq8im2cWtUci3ocKgoMiu6Ln_/view?usp=sharing)

[macOS](https://drive.google.com/file/d/1sWVr8h9wWQd7ciHBWVowdXzFe9xpmPtV/view?usp=sharing)

[Ubuntu](https://drive.google.com/file/d/1SEtkcVp2gMww5DqOXbptsJFtasd9l7xh/view?usp=sharing)

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
