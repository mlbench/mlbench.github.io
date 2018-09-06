---
layout: post
title: Introducing MLBench
author: r_grubenmann, m_jaggi
published: true
tags: [introduction]
excerpt_separator: <!--more-->
---
MLBench is a framework for distributed machine learning. Its purpose is to improve transparency, reproducibility, robustness, and to provide fair performance measures as well as reference implementations, helping adoption of distributed machine learning methods both in industry and in the academic community.

<a href="{{ site.baseurl }}public/images/Dashboard_Index.png" data-lightbox="Dashboard_Index" data-title="The MLBench Dashboard">
  <img src="{{ site.baseurl }}public/images/Dashboard_Index.png" alt="The MLBench Dashboard" style="max-width:80%;"/>
</a>

<!--more-->

MLBench is developed completely open-source and vendor-independent, and has two main goals: 

1. to be an easy-to-use and fair benchmarking suite for algorithms as well as for systems (software frameworks and hardware).
2. to provide re-usable and reliable reference implementations of distributed ML algorithms


### Main Features

MLBench is based on [Kubernetes](https://kubernetes.io/) to ease deployment in a distributed setting, and supports several standard machine-learning frameworks

* **Easy setup**: MLBench can be installed with a single shell command, on public clouds as well as on standard hardware.
* **Convenient Dashboard**: MLBench comes with a dashboard that allows easy access and management for running experiments
    - **Monitoring**: The dashboard allows you to monitor resource usage of all worker nodes participating in experiments
    - **Experiment Setup**: Easily start one of the reference experiments or define your own, specifying resource usage, number of nodes, type of experiment and more.
    - **Visualizations**: Quickly get visualizations of your runs, including quality metrics and resources usage in total and on individual workers
* **Expandable**: We aim to easily allow users to implement and add their own models and algorithms to MLBench, with minimal changes, while still benefiting from all the features of the benchmark environment.
* **Fairness and Reproducibility**: By providing precise specifications of the benchmark ML tasks, metrics as well as reference implementations, MLBench provides fair baselines and improves transparency.
* **Versatile**: We aim to support as many of the platforms, ML frameworks and common machine learning task as possible.

### Community

Our project is open, vendor independent and backed by academic standards, and we highly value contributions from the community

Github: https://github.com/mlbench/

Mailing list: https://groups.google.com/d/forum/mlbench

### Getting Started

Please refer to our [getting-started tutorial]({% post_url 2018-09-10-tutorial %}) on how to set up and start using MLBench.
