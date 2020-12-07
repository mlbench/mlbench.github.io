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
only present the results for the best obtained speedups

</p>

---

|Task                 | Dataset  | Model                 | Aggregation scheme |  Metric Name           | Metric Goal | Framework    |    Baseline    | Best Speedup   | Best Speedup Workers | Description | Implementation | Results |
|:-------------------:|:--------:|:---------------------:|:------------------:|:----------------------:|:-----------:|:------------:|---------------:|:--------------:|:--------------------:|:-----------:|:--------------:|:-------:|
| Image Recognition   | CIFAR10  | ResNet20              | All-Reduce         | Validation Accuracy    |  80%        |  Torch 1.7.0 |                |                |                      |   [details](https://mlbench.readthedocs.io/en/latest/benchmark-tasks.html#task-1-image-classification)|   [code](https://github.com/mlbench/mlbench-benchmarks/tree/develop/pytorch/imagerecognition/cifar10-resnet20-all-reduce)              |         |
|                     |          |                       | PyTorch DDP        |                        |             |  Torch 1.7.0 |                |                |                      |  [details](https://mlbench.readthedocs.io/en/latest/benchmark-tasks.html#task-1-image-classification)| [code](https://github.com/mlbench/mlbench-benchmarks/tree/develop/pytorch/imagerecognition/cifar10-resnet20-distributed-data-parallel)                |         |
|---------------------+----------+-----------------------+--------------------+------------------------+-------------+--------------+----------------+----------------+----------------------+-------------+---------|
| Language Modelling  | Wikitext2| AWD-LSTM              | All-Reduce         | Validation Perplexity  | 70          |  Torch 1.7.0 |                |                |                      | [details](https://mlbench.readthedocs.io/en/latest/benchmark-tasks.html#task-3-language-modelling) |  [code]()               |         |
| Language Modelling  |          | BERT
|---------------------+----------+-----------------------+--------------------+------------------------+-------------+--------------+----------------+----------------+----------------------+----------------+---------|
| Machine Translation | WMT16    | LSTM (GNMT)           | All-Reduce         | Validation BLEU-score  | 24          |  Torch 1.7.0 |                |                |                      | [details](https://mlbench.readthedocs.io/en/latest/benchmark-tasks.html#a-lstm-wmt16-en-de) | [code](https://github.com/mlbench/mlbench-benchmarks/tree/develop/pytorch/nlp/translation/wmt16-gnmt-all-reduce)                |         |
|---------------------+----------+-----------------------+--------------------+------------------------+-------------+--------------+----------------+----------------+----------------------+----------------+---------|
| Machine Translation | WMT17    | Transformer           | All-Reduce         | Validation BLEU-score  | 25          |  Torch 1.7.0 |                |                |                      | [details](https://mlbench.readthedocs.io/en/latest/benchmark-tasks.html#b-transformer-wmt17-en-de) | [code](https://github.com/mlbench/mlbench-benchmarks/tree/develop/pytorch/nlp/translation/wmt17-transformer-all-reduce)                |         |
{:.wide}

