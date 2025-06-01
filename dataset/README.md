TaggerEva Datasets
=========
TaggerEva includes three datasets:
* IDData: 4,862 identifiers (i.e., method names, parameter names and class names) from open-source projects with human annotation
* NLData: 3,000 natural language sentences sampled from [UD-GUM](https://gucorpling.org/gum/).  
* MNTrain: contains 6,165 method names sampled from 10 open-source Java projects and 


The columns are explained below:
1. ID: The number of ordered data item.
2. SEQUENCE: The token sequence. Tokens in both two type of data are splitted by the blank spacing.
3. POS: The corresponding part-of-speech tag sequence of the token sequence.
4. PROJECT (IDData/MNTrain): The source project of the identifier.
5. FILE (IDData/MNTrain): The source file of the identifier.

The dataset is splitted into three parts:

| Data | #Item | #Project |
|----|----|----|
|MNTrain-training|5,588|9|
|MNTrain-dev|577|1|
|Test(IDData)|4,862|9|
|Test(NLData)|3,000|-|

Adoption for taggers
------
For adoption to the special training/testing demand like command line interface, we transform the IDData in several versions:
1. Stanford Format
> on/IN ready/JJ

The dataset has been transform to Stanford Format in the dir "stanford_format".

2. spaCy binary format: In the dir "spacy_format".
   
3. Flair format
> get VB
> 
> id NN

4. Ensemble format
    * TYPE: The return value's type of the method.
    * DECLARATION: Other parts of a method declaration.
