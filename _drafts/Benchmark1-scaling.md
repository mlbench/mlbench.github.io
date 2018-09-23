---
layout: post
title: Scaling Up Nodes
tags: [benchmark, mlbench, scaling]
---

In this post, we show benchmark results on a standard deep learning task (`CIFAR-10`), for distributed training using the standard SGD algorithm on a public cloud, scaling 2,4,8,16,32,64 nodes.

**Table of Contents**
* TOC
{:toc}

## Benchmark Task
Distributed SGD for image classification on the CIFAR-10 dataset.

### ML model

|key | value |
|:----:|:----:|
|model | ResNet 20 with Pre-Activation <a href="#kaiming16identity">[2]</a>|
|dataset | CIFAR-10|
|dataset preprocessing| described in <a href="#kaiming15deep">[1]</a>|
|weight decay| 0.0001|
|#parallel workers (data loading)| 0 |

## Training Algorithm
Standard mini-batch distributed Stochastic Gradient Descent (SGD)

- TODO: describe basic algo first, details later

### Learning rate
There are 3 aspects to consider: `scaling`, `warmup`, `scheduling`

#### Scaling learning rates by number of machines
We use `k` for the number of machines and `b` for minibatch size per worker. 

The `Linear Scaling Rule` in <a href="#goyal2017accurate">[3]</a> states:
>Linear Scaling Rule: When the minibatch size is multiplied by `k`, multiply the learning rate by `k`.

In each iteration, `k*b` samples are trained. If no scaling is used, then 
```python
lr = eta * b
```
where `eta` is learning rate per sample. If we use `Linear Scaling Rule`, then
```python
lr = eta * b * k
```

In <a href="#kaiming15deep">[1]</a> they use learning rate `0.1=k*b*eta=2*128*eta=256*eta` for 2 workers with batch size of 128. So here we choose `eta=0.1/256` 

#### Warmup to scaled learning rates

In <a href="#goyal2017accurate">[3]<a> they use a `warmup` strategy which
>using lower learning rates at the start of training to overcome early optimization difficulties.

In this experiment we start from a learning rate of `eta * b` and linearly increase it to `eta * b * k` with in `5` epochs.

#### Scheduling
We reduce the learning rate by `0.1` at 82-th, 109-th epoch for all workers.
This is similar to <a href="#kaiming15deep">[1]</a> which reduces at 32k-th, 48k-th iterations.

### Dataset

The datasets are partitioned by nodes. Each worker only has access to part of the data.

### Training Algorithm Summary

|key | value |
|:----:|:----:|
|minibatch size per worker (b)| 32 |
|number of workers (k)| 2, 4, 8, 16, 32, 64 |
|learning rate per sample (eta)| 0.1 / 256|
|scaled learning rate | k * b * eta |
|train epochs | 164 |
|learning rate schedule | Starting from reference learning rate and reduce by 1/10 at the 82-th, 109-th epoch <a href="#kaiming15deep">[1]</a>.|
|warmup| warmup lr for the first 5 epochs and starts with `eta * b`|
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
|limits.cpu| 2500m|
|limits.workers| 1|
|limits.bandwidth| about 7.5Gbip/s|

## Results
* Epochs to Top-1 Validation Accuracy
<a href="{{ site.baseurl }}public/images/scaling-epoch-prec1.png" data-lightbox="Run" data-title="Validation Accuracy @ 1">
  <img src="{{ site.baseurl }}public/images/scaling-epoch-prec1.png" alt="Validation Accuracy @ 1" style="max-width:80%;"/>
</a>
* Time to Top-1 Validation Accuracy
<a href="{{ site.baseurl }}public/images/scaling-time-prec1.png" data-lightbox="Run" data-title="Validation Accuracy @ 1">
  <img src="{{ site.baseurl }}public/images/scaling-time-prec1.png" alt="Validation Accuracy @ 1" style="max-width:80%;"/>
</a>


## Conclusion
TBD

<!-- ### Cluster Management -->
<!-- Scripts to create/scale/delete cluster. -->

<!-- {% gist 23361aea5fe252570496acc7da4fb599 %} -->



## Reference
<p id="kaiming15deep">[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. Deep Residual Learning for Image Recognition. arXiv:1512.03385</p>
<p id="kaiming16identity">[2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. Identity Mappings in Deep Residual Networks. arXiv: 1603.05027</p>
<p id="goyal2017accurate">[3] Goyal, Priya, et al. Accurate, large minibatch SGD: training imagenet in 1 hour.</p>

