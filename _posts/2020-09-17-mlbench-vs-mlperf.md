---
layout: post
title: Comparison of MLBench and MLPerf
author: m_milenkoski
published: true
tags: [mlperf, introduction]
excerpt_separator: <!--more-->
---

[MLPerf](https://mlperf.org/) is a broad benchmark suite for measuring the performance of machine learning (ML) software frameworks, ML hardware platforms and ML cloud platforms. 

In this post, we will highlight the main differences between MLBench and MLPerf. 

<!--more-->


## Key Advantages of MLBench

- Reference implementations for distributed execution instead of only single-node execution as in MLPerf
- Improved ease-of-use of benchmarks through the CLI or Dashboard instead of the manual setup in MLPerf
- Public cloud support (currently Google Cloud, AWS and local deployment)
- Full PyTorch support
- Fine-grained metrics for the execution time of all relevant components
- Performance comparisons between different communication backends
- Additional light goal for each task, for quick iterations
- Showing scaling efficiency when increasing number of nodes
- Providing many different optimization algorithms and communication patterns
- Support for easier local development of distributed machine learning methods (via kubernetes in docker)

## Results reporting

Both MLBench and MLPerf use end-to-end time to accuracy. However, MLBench reports how much time was used on communication and how much on computing, while MLPerf does not. Furthermore, MLBench reports more fine grained results in comparison to MLPerf. While MLPerf allows for distributed training, it does not distinguish the results obtained from single node and multi-node training. Moreover, MLPerf does not distinguish between single GPU and multiple GPUs per node. While the number of nodes and GPUs per node are reported, there does not seem to be any fine grained reporting on the amount of time spent on communication and computation. MLPerf reports only one number for all possible scenarios - time to accuracy. For this reason, MLPerf can not accurately show the effects of scaling the number of nodes or GPUs. They are also not able to pinpoint the reason for an improved or decreased performance of a model because their reported results are not fine grained. On the other hand, MLBench is fully focused on distributed training, can show the effects of scaling, can identify the bottlenecks in the model performance and can accurately show the effects of the model hyperparameters on different parts like communication and computation. In this way, MLBench offers a much more powerful and versatile benchmarking suite than the one offered by MLPerf. 

## Hyperparameter tuning

MLPerf restricts the set of hyperparameters that can be tuned. It also allows users to borrow hyperparameters from others. MLBench currently provides exact values for all hyperparameters.

## Benchmark Suites

<table>
<thead>
  <tr>
    <th rowspan="2">Benchmark</th>
    <th colspan="2">Dataset</th>
    <th colspan="2">Quality Target</th>
    <th colspan="2">Reference Implementation Model</th>
    <th colspan="2">Frameworks</th>
  </tr>
  <tr>
    <td>MLBench</td>
    <td>MLPerf</td>
    <td>MLBench</td>
    <td>MLPerf</td>
    <td>MLBench</td>
    <td>MLPerf</td>
    <td>MLBench</td>
    <td>MLPerf</td>
  </tr>
</thead>
<tbody>
  <tr>
    <td>Image classification</td>
    <td>CIFAR10 (32x32)</td>
    <td>/</td>
    <td>80% Top-1 Accuracy</td>
    <td>/</td>
    <td>ResNet-20</td>
    <td>/</td>
    <td>PyTorch, Tensorflow</td>
    <td>/</td>
  </tr>
  <tr>
    <td>Image classification</td>
    <td colspan="2">ImageNet (224x224)</td>
    <td>TODO</td>
    <td>75.9% Top-1 Accuracy</td>
    <td>TODO</td>
    <td>Resnet-50 v1.5</td>
    <td>TODO</td>
    <td>MXNet, Tensorflow</td>
  </tr>
  <tr>
    <td>Object detection (light weight)</td>
    <td>/</td>
    <td>COCO 2017</td>
    <td>/</td>
    <td>23% mAP</td>
    <td>/</td>
    <td>SSD-ResNet34</td>
    <td>/</td>
    <td>Tensorflow, PyTorch</td>
  </tr>
  <tr>
    <td>Object detection (heavy weight)</td>
    <td>/</td>
    <td>COCO 2017</td>
    <td>/</td>
    <td>0.377 Box min AP, 0.339 Mask min AP</td>
    <td>/</td>
    <td>Mask R-CNN</td>
    <td>/</td>
    <td>Tensorflow, PyTorch</td>
  </tr>
  <tr>
    <td>Language Modelling</td>
    <td>Wikitext2</td>
    <td>/</td>
    <td>Perplexity &lt;= 50</td>
    <td>/</td>
    <td>RNN-LM</td>
    <td>/</td>
    <td>PyTorch</td>
    <td>/</td>
  </tr>
  <tr>
    <td>Translation (recurrent)</td>
    <td>WMT16 EN-DE</td>
    <td>WMT English-German</td>
    <td colspan="2">24.0 BLEU</td>
    <td colspan="2">GNMT</td>
    <td>PyTorch</td>
    <td>Tensorflow, PyTorch</td>
  </tr>
  <tr>
    <td>Translation (non-recurrent)</td>
    <td>WMT17 EN-DE</td>
    <td>WMT English-German</td>
    <td colspan="2">25.0 BLEU</td>
    <td colspan="2">Transformer</td>
    <td>PyTorch</td>
    <td>Tensorflow, PyTorch</td>
  </tr>
  <tr>
    <td>Recommendation</td>
    <td>/</td>
    <td>Undergoing modification</td>
    <td>/</td>
    <td></td>
    <td>/</td>
    <td></td>
    <td>/</td>
    <td></td>
  </tr>
  <tr>
    <td>Reinforcement learning</td>
    <td>/</td>
    <td>N/A</td>
    <td>/</td>
    <td>Pre-trained checkpoint</td>
    <td>/</td>
    <td>Mini Go</td>
    <td>/</td>
    <td>Tensorflow</td>
  </tr>
</tbody>
</table>
