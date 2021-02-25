---
layout: post
title: Benchmarking Communication Backend
tags: [mpi]
---

In this experiment, we use gcloud Kubernetes cluster. For nodes with 1 or 2 gpus we use `n1-standard-4` type machine; for nodes with 4 GPUs, `n1-standard-8` type machien is used; for nodes with 8 GPUs, `n1-standard-16` type machine is used. K80 GPUs are used for all of the experiments. 

We use the `mlbench/mlbench_worker:latest` as the worker image which contains `openmpi`, `pytorch` and `NCCL`.

## P2P Communication
In this experiment, we compare MPI P2P communication for 

- Sending/receiving vectors on cpu/gpu of two ndoes;
- Sending/receiving vectors between two GPUs on one node with or without IPC option.

CUDA Inter-Process Communication (IPC) improves communication between GPUs on the same node. In openmpi, one can use `--mca btl_smcuda_use_cuda_ipc` to turn on/off this functionality. We demostrate the influence of CUDA-IPC by sending/receiving a vector on a node with two GPUs. 

### Results
<a href="{{ site.baseurl }}public/images/blog/drafts/mpi-speed-p2p.png" data-lightbox="Run" data-title="MPI Speed P2P">
  <img src="{{ site.baseurl }}public/images/blog/drafts/mpi-speed-p2p.png" alt="MPI Speed P2P" style="max-width:80%;"/>
</a>

- The P2P communication between two nodes is bounded by network bandwidth (`7.5 Gbit/s` measured by `iperf`). Communicating large vectors on CPU/GPU have similar throughput.
- Enabling CUDA-IPC option gives considerable performance gain. However the connection between GPUs is `PHB` (not NVLink) which would be limited by the bandwidth of PCIe (like `8 GB/s`).

## Collective Communication

In this experiment, we compare the performance of all reduce NCCL between and MPI in:

* 8 nodes with 1 GPU per node
* 4 nodes with 2 GPUs per node
* 2 nodes with 4 GPUs per node
* 1 node with 8 GPUs

We use PyTorch distributed package to all reduce float32 vectors with 10 to 10^9 elements (~4GB). All experiments are repeated for 100 times.

### Environment
The connection between GPUs are PHB which traverss PCIe as well as a PCIe Host Bridge (typically the CPU), see appendix. NCCL's performance can be improved if NVLink high-speed interconnect is available.

### Results
The results of experiments are shown below. Note that the bandwidth here is calculated by dividing the vector size by the time it spent. The actual bandwidth depends on the implementation of all reduce.

<a href="{{ site.baseurl }}public/images/blog/drafts/mpi-speed-collective.png" data-lightbox="Run" data-title="MPI Speed Collective">
  <img src="{{ site.baseurl }}public/images/blog/drafts/mpi-speed-collective.png" alt="MPI Speed Collective" style="max-width:80%;"/>
</a>

- The NCCL all reduce does not give better performance when the GPU per machine is 1 or 2.
- If the vector is large and there is no inter-node communication (1 node 8 GPUs), NCCL also outperforms MPI.

## Conclusion
To compare performance of distributed training, it is better to include

- the configuration of MPI parameters;
- the topology of GPUs (pairwise bandwidth, etc);

It is also interesting to see that NCCL out-performs MPI even there is only 1 node with 8 gpus.

## Appendix
Code can be found [here](https://github.com/LiamHe/mlbench_benchmarking/tree/master/00-communication-backend).

##### IPC warning
Note that when `--mca btl_smcuda_use_cuda_ipc` option is enabled, there will be a warning after the program finishes

    --------------------------------------------------------------------------
    The call to cuIpcCloseMemHandle failed. This is a warning and the program
    will continue to run.
      cuIpcCloseMemHandle return value:   4
      address: 0x2113a0000
    Check the cuda.h file for what the return value means. Perhaps a reboot
    of the node will clear the problem.
    --------------------------------------------------------------------------
    [rel-mlbench-worker-0:02668] Sleep on 2668

After that the process will sleep for 20 seconds and then exit normally.


##### To make sure NCCL is installed
```python
>>> import torch
>>> print(torch.cuda.nccl.is_available())
True
>>> print(torch.cuda.nccl.version())
2005
```

##### To know the topology of GPUs
```bash
nvidia-smi topo --matrix
```
<pre>
    GPU0    GPU1    GPU2    GPU3    GPU4    GPU5    GPU6    GPU7    CPU Affinity
GPU0     X  PHB PHB PHB PHB PHB PHB PHB 0-15
GPU1    PHB  X  PHB PHB PHB PHB PHB PHB 0-15
GPU2    PHB PHB  X  PHB PHB PHB PHB PHB 0-15
GPU3    PHB PHB PHB  X  PHB PHB PHB PHB 0-15
GPU4    PHB PHB PHB PHB  X  PHB PHB PHB 0-15
GPU5    PHB PHB PHB PHB PHB  X  PHB PHB 0-15
GPU6    PHB PHB PHB PHB PHB PHB  X  PHB 0-15
GPU7    PHB PHB PHB PHB PHB PHB PHB  X  0-15

Legend:

  X    = Self
  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
  NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges within a NUMA node
  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
  PXB  = Connection traversing multiple PCIe switches (without traversing the PCIe Host Bridge)
  PIX  = Connection traversing a single PCIe switch
  NV#  = Connection traversing a bonded set of # NVLinks
</pre>
Topology Test : `/usr/local/cuda/samples/1_Utilities/topologyQuery/topologyQuery`
<pre>
GPU0 <-> GPU1:
  * Atomic Supported: no
  * Perf Rank: 0
...
</pre>

## Reference
- [WRITING DISTRIBUTED APPLICATIONS WITH PYTORCH](https://pytorch.org/tutorials/intermediate/dist_tuto.html)
- [Distributed communication package - torch.distributed](https://pytorch.org/docs/stable/distributed.html)
- [How to properly use distributed pytorch with infiniband support](https://discuss.pytorch.org/t/how-to-properly-use-distributed-pytorch-with-infiniband-support/10161)