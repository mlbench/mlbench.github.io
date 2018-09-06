---
layout: post
title: Introducing MLBench
author: r_grubenmann
published: true
tags: [introduction]
excerpt_separator: <!--more-->
---
MLBench is a benchmarking framework for comparing distributed machine learning algorithms on different platforms, infrastructure and topology.

<a href="{{ site.baseurl }}public/images/Dashboard_Index.png" data-lightbox="Dashboard_Index" data-title="The MLBench Dashboard">
  <img src="{{ site.baseurl }}public/images/Dashboard_Index.png" alt="The MLBench Dashboard" style="max-width:80%;"/>
</a>

<!--more-->

### Why MLBench
MLBench aims to be a robust, fair and easy to use benchmark that allows for independent evaluation of distributed machine learning algorithms.
It is based on [Kubernetes](https://kubernetes.io/) to ease deployment in a distributed setting and uses [Helm](https://helm.sh/) to make installation as simple as possible. Kubernetes is turning into the de-facto standard for installing and managing distributed applications and is supported by all major cloud providers as well as being a popular solution for data centers and private clouds.

Our benchmarks are independent of any vendor and aim to be as fair and comparable as possible, allowing for a meaningful discussion and comparison of results across a multitude of approaches.
We hope that this will help to further research and innovation as well as allowing developers and suppliers of distributed machine learning service to make well-founded and reliable decisions for their projects.

### Main Features

* **Easy to setup**: Kubernetes combined with Helm means MLBench can be installed with a single shell command.
* **Convenient Dashboard**: MLBench comes with a dashboard that allows easy access and management for running experiments
    - **Monitoring**: The dashboard allows you to monitor resource usage of all worker nodes participating in experiments
    - **Experiment Setup**: Easily start one of the reference experiments or define your own, specifying resource usage, number of nodes, type of experiment and more.
    - **Visualizations**: Quickly get visualizations of your runs, of resources used and of metrics on individual workers
* **Expandable**: We aim to easily allow users to implement and add their own models to MLBench, with minimal changes, while still benefiting from all the features we provide.
* **Fair and Reproducible**: We aim to be as fair as possible, providing independent baselines without any unfair tricks or hacks.
* **Versatile**: We aim to support as many platforms, frameworks and machine learning domains as possible.

### Getting Started

Please refer to our [in-depth tutorial]({% post_url 2018-09-10-tutorial %}) to see how to set up and use MLBench.