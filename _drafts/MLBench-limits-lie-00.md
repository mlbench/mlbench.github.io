---
layout: post
title: Limiting Resources for Benchmarking
tags: [benchmark, mlbench]
---

The resources can be limited here. 

The documentation can be seen [here](https://mlbench.readthedocs.io/en/develop/installation.html#helm-chart-values).

---
## Example `n1-standard-4`

First create a cluster of two `n1-standard-4` instances with `limits.cpu=1000m`

### master/worker-0 node

<a href="{{ site.baseurl }}public/images/lie-00-resource-node1.png" data-lightbox="Dashboard_Index" data-title="lie-00-resource-node1">
  <img src="{{ site.baseurl }}public/images/lie-00-resource-node1.png" alt="lie-00-resource-node1" style="max-width:100%;"/>
</a>

Only 2 pods are for mlbench: `release1-mlbench-master-6448bfb454-sxm2l` (`100m` CPU) and `release1-mlbench-worker-0` (`1000m` CPU). The rest of pods request `1161m` of CPU and `750MB` memory.

The summary of resources on this node is (requests `2261m` CPU in total )

<a href="{{ site.baseurl }}public/images/lie-00-resource-node1-summary.png" data-lightbox="Dashboard_Index" data-title="The MLBench Dashboard">
  <img src="{{ site.baseurl }}public/images/lie-00-resource-node1-summary.png" alt="The MLBench Dashboard" style="max-width:100%;"/>
</a>

### worker-1 node
On worker-1 node, there are much less pods.
<a href="{{ site.baseurl }}public/images/lie-00-resource-node2.png" data-lightbox="Dashboard_Index" data-title="lie-00-resource-node2">
  <img src="{{ site.baseurl }}public/images/lie-00-resource-node2.png" alt="lie-00-resource-node2" style="max-width:100%;"/>
</a>

So the amount of resources available is limited to the master node. In the previous setting we can allocate at most `3920-1161-100=2659m` for each worker.

---
## Choosing `limits.cpu` for worker
1. Create a cluster (Note: do not use `--preemptible` for jobs running for more than 24 hours)
2. Find out the resources caused by `kubernetes` related pods on master node.
3. Change the `limits.cpu` so that `limits.cpu + mlbench-master.cpu + kubernetes.cpu < allocatable CPUs`.
4. Install the helm charts

---
## Bandwidth
The bandwidth is limited by the neighborhood of the number of.

---
## Accelerator
TBD