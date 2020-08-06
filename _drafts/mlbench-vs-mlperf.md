---
layout: post
title: Comparison of MLBench and MLPerf
---

MLPerf is a broad benchmark suite for measuring the performance of machine learning (ML) software frameworks, ML hardware platforms and ML cloud platforms. 

In this post, we will highlight the main differences between MLBench and MLPerf. 

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

### MLBench

|   Task  	|           Benchmark          	|       Dataset      	|   Quality Target   	| Reference Implementation Model 	|      Frameworks     	|
|:-------:	|:----------------------------:	|:------------------:	|:------------------:	|:------------------------------:	|:-------------------:	|
| Task 1a 	|     Image classification     	|   CIFAR10 (32x32)  	| 80% Top-1 Accuracy 	|            ResNet-20           	| PyTorch, Tensorflow 	|
| Task 1b 	|     Image classification     	| ImageNet (224x224) 	|        TODO        	|              TODO              	|         TODO        	|
| Task 3a 	|      Language Modelling      	|      Wikitext2     	|  Perplexity <= 50  	|             RNN-LM             	|       PyTorch       	|
| Task 4a 	|    Translation (recurrent)   	|     WMT16 EN-DE    	|      24.0 BLEU     	|              GNMT              	|       PyTorch       	|
| Task 4b 	| Translation (non-recurrent)  	|     WMT17 EN-DE    	|      25.0 BLEU     	|           Transformer          	|       PyTorch       	|

### MLPerf

|            Benchmark            	|         Dataset         	|            Quality Target           	| Reference Implementation Model 	|      Frameworks     	|
|:-------------------------------:	|:-----------------------:	|:-----------------------------------:	|:------------------------------:	|:-------------------:	|
|       Image classification      	|    ImageNet (224x224)   	|         75.9% Top-1 Accuracy        	|         Resnet-50 v1.5         	|  MXNet, Tensorflow  	|
| Object detection (light weight) 	|        COCO 2017        	|               23% mAP               	|          SSD-ResNet34          	| Tensorflow, PyTorch 	|
| Object detection (heavy weight) 	|        COCO 2017        	| 0.377 Box min AP, 0.339 Mask min AP 	|           Mask R-CNN           	| Tensorflow, PyTorch 	|
|     Translation (recurrent)     	|    WMT English-German   	|              24.0 BLEU              	|              GNMT              	| Tensorflow, PyTorch 	|
|   Translation (non-recurrent)   	|    WMT English-German   	|              25.0 BLEU              	|           Transformer          	| Tensorflow, PyTorch 	|
|          Recommendation         	| Undergoing modification 	|                                     	|                                	|                     	|
|      Reinforcement learning     	|           N/A           	|        Pre-trained checkpoint       	|             Mini Go            	|      Tensorflow     	|

