# RQ1
Evaluation of six off-the-shelf taggers on IDData/NLData.
## Setup
1. Change the mode(all/method/args/class/nl) in ```config.py```;
2. (Optional) run the following script and get  ```evaluation_{mode}.csv``` as results.
```shell
python evaluation.py
```

> ```opennlp_output``` contains the output results form OpenNLP CLI.

3. Get the metric value from outputs. Run the ```test.py```
```shell
python test.py
```