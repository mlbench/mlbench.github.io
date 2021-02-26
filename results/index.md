---
layout: page
notitle: true
title: Results
datatable: true
---

# MLBench Official Training Results
---

<p>
Here we present the official training results obtained by running the benchmark tasks.
It provides a reference and comparison point for different implementations.
</p>


<p>
The table below provides a comparison table for results. Times are in seconds. The baseline represents the 1 worker case, and we
only present the results for the best obtained speedups.

For more detailed results for each run, please check the documentation.

</p>

---

|Task                 | Dataset  | Model                 | Aggregation scheme |  Metric Name           | Metric Goal | Framework    |    Baseline    | Best Speedup   | Best Speedup Workers | Description | Implementation | Results |
|:-------------------:|:--------:|:---------------------:|:------------------:|:----------------------:|:-----------:|:------------:|:--------------:|:--------------:|:--------------------:|:-----------:|:--------------:|:-------:|
| Image Recognition   | CIFAR10  | ResNet20              | All-Reduce         | Validation Accuracy    |  80%        |  Torch 1.7.0 |    171.15s     |   6.6 (total), 11.9 (compute) |  16   |   [details](https://mlbench.readthedocs.io/en/latest/benchmark-tasks.html#task-1-image-classification)|   [code](https://github.com/mlbench/mlbench-benchmarks/tree/develop/pytorch/imagerecognition/cifar10-resnet20-all-reduce)              |   [results](https://mlbench.readthedocs.io/en/latest/benchmark-tasks.html#id17)      |
|                     |          |                       | PyTorch DDP        |                        |             |  Torch 1.7.0 |    182.81s     | 4.4            |           8          |  [details](https://mlbench.readthedocs.io/en/latest/benchmark-tasks.html#task-1-image-classification)| [code](https://github.com/mlbench/mlbench-benchmarks/tree/develop/pytorch/imagerecognition/cifar10-resnet20-distributed-data-parallel)                |         |
|---------------------+----------+-----------------------+--------------------+------------------------+-------------+--------------+----------------+----------------+----------------------+-------------+---------|
| Language Modelling  | Wikitext2| AWD-LSTM              | All-Reduce         | Validation Perplexity  | 70          |  Torch 1.7.0 |    87,401.33s  |                |                      | [details](https://mlbench.readthedocs.io/en/latest/benchmark-tasks.html#task-3-language-modelling) |  [code]()               |         |
| Language Modelling  |          | BERT
|---------------------+----------+-----------------------+--------------------+------------------------+-------------+--------------+----------------+----------------+----------------------+----------------+---------|
| Machine Translation | WMT16    | LSTM (GNMT)           | All-Reduce         | Validation BLEU-score  | 24          |  Torch 1.7.0 |    65,206.62s   | 2.8 (total), 15.8 (compute)          |      16              | [details](https://mlbench.readthedocs.io/en/latest/benchmark-tasks.html#a-lstm-wmt16-en-de) | [code](https://github.com/mlbench/mlbench-benchmarks/tree/develop/pytorch/nlp/translation/wmt16-gnmt-all-reduce)                |   [results](https://mlbench.readthedocs.io/en/latest/benchmark-tasks.html#id30)      |
|---------------------+----------+-----------------------+--------------------+------------------------+-------------+--------------+----------------+----------------+----------------------+----------------+---------|
| Machine Translation | WMT17    | Transformer           | All-Reduce         | Validation BLEU-score  | 25          |  Torch 1.7.0 |    37,594.21s   | 3.0 (total), 18.3 (compute)          |      16              | [details](https://mlbench.readthedocs.io/en/latest/benchmark-tasks.html#b-transformer-wmt17-en-de) | [code](https://github.com/mlbench/mlbench-benchmarks/tree/develop/pytorch/nlp/translation/wmt17-transformer-all-reduce)                |    [results](https://mlbench.readthedocs.io/en/latest/benchmark-tasks.html#id32)     |
{:.wide}

