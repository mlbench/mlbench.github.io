---
layout: post
title: MLBench [1] -- Scaling up nodes (DRAFT)
tags: [benchmark, mlbench, scaling]
---

In this post, we show benchmark results on a standard deep learning task (`CIFAR-10`), for distributed training using the standard SGD algorithm on a public cloud, scaling 2,4,8,16,32,64 nodes.


## Benchmark Task
Distributed SGD for image classification on the CIFAR-10 dataset.

### ML model

|key | value |
|:----:|:----:|
|model | ResNet 20 with Pre-Activation [^fnkaiming16identity]|
|dataset | CIFAR-10|
|dataset preprocessing| described in [^fnkaiming15deep]|
|weight decay| 0.0001|
|#parallel workers (data loading)| 0 |

## Training Algorithm
Standard mini-batch distributed Stochastic Gradient Descent (SGD)

- TODO: describe basic algo first, details later

|key | value |
|:----:|:----:|
|minibatch size per worker (b)| 32 |
|number of workers (k)| 2, 4, 8, 16, 32, 64 |
|learning rate per sample| 0.1 / 256|
|scaled learning rate | k * b * (learning rate per sample) |
|train epochs | 164 |
|learning rate schedule | Starting from reference learning rate and reduce by 1/10 at the 82-th, 109-th epoch [^fnkaiming15deep].|
|warmup| warmup lr for the first 5 epochs and starts with 0|
|momentum | 0.9|
|nesterov | True|

## Hardware / System

_Note: This preliminary results uses Google cloud kubernetes instances. We will work to obtain results for other clouds and instances soon. Since we do not have enough quotas for 64 GPUs, only CPUs are used._

|key|value|
|:---:|:---:|
|communication backend| MPI|
|machine type | [n1-standard-4](https://cloud.google.com/compute/pricing) : 4 vCPUs, 15GB memory (note if the job last more than 24 hours, then do not choose preemptible)|
|accelerator| None|
|instance disk size|  50 GB|
|instance disk type| pd-standard|
|worker pod per node| 1 |

### Helm Chart Values
The definitions helm chart values can be found [here](https://mlbench.readthedocs.io/en/develop/installation.html#helm-chart-values).

|key|value|
|:---:|:---:|
|limits.cpu| 1000m|
|limits.maximumWorkers| 1|
|limits.bandwidth| 10000|

### Detailed Experiment Configurations

**Linear scaling rule**

The `reference learning rate` in the previous table comes from *Linear Scaling Rule* [^goyal2017accurate]. In [^fnkaiming15deep] they use learning rate 0.1 for k * n=2 * 128 = 256, so using the `reference learning rate` above can recover their settings.

**Version of source code**
commit 

### Cluster Management
Scripts to create/scale/delete cluster.

{% gist 23361aea5fe252570496acc7da4fb599 %}

## Results
* Time to Top-1 Validation Error
* Time to Train/Validation Loss

TODO: depending number of nodes

## Conclusion
TBD

## Reference
[^fnkaiming15deep]: Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. Deep Residual Learning for Image Recognition. arXiv:1512.03385
[^fnkaiming16identity]: Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. Identity Mappings in Deep Residual Networks. arXiv: 1603.05027
[^goyal2017accurate]: Goyal, Priya, et al. Accurate, large minibatch SGD: training imagenet in 1 hour.
