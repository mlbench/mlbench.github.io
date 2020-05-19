---
layout: post
title: "Tutorial: Using the MLBench Commandline Interface"
author: r_grubenmann
published: true
tags: [tutorial,guide]
excerpt_separator: <!--more-->
---

We recently released MLBench version 2.1.0, which contains a new commandline interface, making it even easier to run our benchmarks.

In this post we'll introduce the CLI and show you how easy it is to get it up and running.

<!--more-->

**Please beware any costs that might be incurred by running this tutorial on the Google cloud. Usually costs should only be on the order of 5-10USD. We don't take any responsibility costs incurred**

Install the [mlbench-core](https://github.com/mlbench/mlbench-core/tree/master) python package by running:

```shell
$ pip install mlbench-core
```

After installation, mlbench is usable by calling the ``mlbench`` command.

To create a new Google cloud cluster, simply run (this might take a couple of minutes):

```shell
$ mlbench create-cluster gcloud 3 my-cluster
[...]
MLBench successfully deployed
```

This creates a cluster with 3 nodes called ``my-cluster-3`` and sets up the mlbench deployment in that cluster. Note that the number of nodes should always be 1 higher than the maximum number of workers you want to run.

To start an experiment, simpy run:

```shell
$ mlbench run my-run 2

Benchmark:

[0] PyTorch Cifar-10 ResNet-20 Open-MPI
[1] PyTorch Cifar-10 ResNet-20 Open-MPI (SCaling LR)
[2] PyTorch Linear Logistic Regrssion Open-MPI
[3] Tensorflow Cifar-10 ResNet-20 Open-MPI
[4] Custom Image

Selection [0]: 1

[...]

Run started with name my-run-2
```

You will be prompted to select the benchmark image you want to run (or to specify a custom image). Afterwards, a new benchmark run will be started in the cluster with 2 workers.

You can also start multiple runs at the same time, which will be scheduled as nodes become available:

```shell
$ mlbench run my-run 2 4 8 16

Benchmark:

[0] PyTorch Cifar-10 ResNet-20 Open-MPI
[1] PyTorch Cifar-10 ResNet-20 Open-MPI (SCaling LR)
[2] PyTorch Linear Logistic Regrssion Open-MPI
[3] Tensorflow Cifar-10 ResNet-20 Open-MPI
[4] Custom Image

Selection [0]: 1

[...]

Run started with name my-run-2
Run started with name my-run-4
Run started with name my-run-8
Run started with name my-run-16
```

which would start runs with 2, 4, 8 and 16 workers, respectively.

To see the status of a run, execute:

```shell
$ mlbench status my-run-2
[...]
id      name    created_at            finished_at state
---     ------  -----------            ----------- -----
1       my-run-2 2019-11-11T13:35:06              started
No Validation Loss Data yet
No Validation Precision Data yet
```

After the first round of validation, this command also outputs the current validation loss and precision.

To download the results of a current or finished run, use:

```shell
$ mlbench download my-run-2
```

which will download all the metrics of the run as a zip file. This file also contains the official benchmark result once the run finishes, in the form of the ``official_result.txt``.

You can also access all the information of the run in the dashboard. To get the dashboard URL, simply run:

```shell
$ mlbench get-dashboard-url
[...]
http://34.76.223.123:32535
```

Don't forget to delete the cluster once you're done!

```shell
$ mlbench delete-cluster gcloud my-cluster-3
[...]
```

**NOTE**: if you created a cluster in a non-default zone using the `-z` flag, 
you also need to delete it by passing the same flag and argument to `mlbench delete-cluster`.

```shell
# create cluster in europe-west2-b (non-default)
$ mlbench create-cluster gcloud -z europe-west2-b 3 my-cluster

# delete cluster
$ mlbench delete-cluster gcloud -z europe-west2-b my-cluster-3
```
