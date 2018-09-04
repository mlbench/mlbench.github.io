---
layout: post
title: MLBench [1] -- Scaling up nodes (DRAFT)
tags: [benchmark, mlbench, scaling]
---

In this post, we benchmark MLBench Framework on Google Cloud Kubernetes Instances on 2,4,8,16,32,64 nodes.

_Note: since we do not have enough quotas for 64 gpus, only cpus are used._

**questions**
A few questions to be answered:
1. where to store github gist (for the moment I keep it to my account)
2. how do we keep the experiment results? (where and in which format)
3. shall we keep preprocessing scripts as gist?
4. optimize the Helm Chart Values for cpu and bandwidth.


## Settings
### Hardware/Platform

|key|value|
|:---:|:---:|
|machine type | [preemptible n1-standard-4](https://cloud.google.com/compute/pricing) |
|accelerator| None|
|instance disk size|  50 GB|
|worker pod per node| 1 |

### Helm Chart Values

|key|value|
|:---:|:---:|
|limits.cpu| 1000m|
|limits.maximumWorkers| 1|
|limits.bandwidth| 100|

### Experiment Configurations

|key | value |
|:----:|:----:|
|model | ResNet 20 with Pre Activation [^fnkaiming16identity]|
|dataset | CIFAR-10|
|dataset preprocessing| described in [^fnkaiming15deep]|
|minibatch per worker (n)| 32 |
|number of workers (k)| 2, 4, 8, 16, 32, 64 |
|reference learning rate | 0.1 * k * n/256|
|train epochs | 164 |
|LR schedule | Starting from reference learning rate and reduce by 1/10 at the 82-th, 109-th epoch [^fnkaiming15deep]. Gradual warmup for the first 5 epochs.|
|weight decay| 0.0001|
|momentum | 0.9|
|#parallel workers (data loading)| 2 |
|communication backend| MPI|
|criterion| CrossEntropyLoss|


**Linear scaling rule**

The `reference learning rate` in the previous table comes from *Linear Scaling Rule* [^goyal2017accurate]. In [^fnkaiming15deep] they use learning rate 0.1 for $k*n=2*128=256$, so using the `reference learning rate` above can recover their settings.


## Cluster Management
Scripts to create/scale/delete cluster.

{% gist 23361aea5fe252570496acc7da4fb599 %}

## Results
* Time to Top-1 Validation Error
* Time to Train/Validation Loss

## Conclusion
TBD

## Reference
[^fnkaiming15deep]: Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. Deep Residual Learning for Image Recognition. arXiv:1512.03385
[^fnkaiming16identity]: Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. Identity Mappings in Deep Residual Networks. arXiv: 1603.05027
[^goyal2017accurate]: Goyal, Priya, et al. Accurate, large minibatch SGD: training imagenet in 1 hour.
