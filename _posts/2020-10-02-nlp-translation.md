---
layout: post
title: NLP Translation tasks, results discussion
author: e_hoelzl
published: true
tags: [performance, results]
excerpt_separator: <!--more-->
---

During the past years, Natural Language Processing (NLP) has gained a lot of interest in the Machine Learning community. 
This increasing attention towards the subject may come from the fascination of teaching machines to understand and assimilate human language, 
and use them as tools to complement and facilitate our everyday lives. 

Machine translation is one branch of NLP, and consists of having automated model capable of translating text from one language to another in a few seconds. 

In this blog post, we analyze the performance increase brought by distribution for two different Machine Translation models: an LSTM variant (GNMT) and an attention 
based model (Transformer).

<!--more-->

Those models present two main limitations that makes training very time consuming:
 - They need millions of data points to reach acceptable performance.
 - Models are quite large, and computations take significantly more time.

Each of those problems can be solved using distribution:
 - Distribute the data on multiple machines (data-parallel)
 - Distributed computations on multiple cores (compute-parallel)
 
However the second approach requires the models to be parallelizable. We have only used data-parallel distribution for those tasks, but hope to combine it
with parallel computation in the future.


## Models

First let's have a quick look at the models' architectures to understand the scale.

### LSTM
The LSTM variant we implemented was designed by Google {% cite gnmt %} and is called the Google Neural Machine translation. 
The architecture is shown in the figure below

<a href="{{ site.baseurl }}public/images/blog/2020-10-02-nlp-translation/gnmt.png" data-lightbox="gnmt_architecture" data-title="GNMT Architecture">
  ![test]({{ site.baseurl }}public/images/blog/2020-10-02-nlp-translation/gnmt.png)
  *GNMT Architecture*
</a>

Left side is the Encoder network, right side is the Decoder, connected via the attention module. 
The first encoder LSTM layer is bi-directional, and others are uni-directional. 
Residual connections start from the layer third from the bottom in the encoder and decoder.

This model follows the sequence-to-sequence learning framework, and uses stacked residual LSTM connections in the encoder and decoder modules. 
The residual connections allow for  deeper  stacked  LSTM  layers,  as  without  residuals,  the  stack  typically  suffer  from vanishing/exploding
gradients when too many layers are used.
The attention module is based on the one described in {% cite bahdanau2014neural %}

In our implementation, the encoder and decoder have each 4 stacked LSTM layers with residual connections, and hidden sizes of 1024. 
This gives a model with a total of 160,671,297 trainable parameters. 

### Transformer

This model was first published in {% cite attention %}, and aims at completely disregarding recurrence and relying entirely on self-attention 
mechanisms to perform sequence modelling and translation problems. 
Such a structure allows for better parallelization of training on multiple GPUs, and can reach significantly better performance than comparable models.

Transformer uses Multi-Head attention mechanisms: instead of computing the attention once, it runs through the scaled dot-product attention multiple times
in parallel.

The figure below shows an overview of the architecture

<a href="{{ site.baseurl }}public/images/blog/2020-10-02-nlp-translation/transformer.png" data-lightbox="transformer_architecture" data-title="Transformer Architecture">
  ![test]({{ site.baseurl }}public/images/blog/2020-10-02-nlp-translation/transformer.png)
  *Transformer Architecture*
</a>

Our implementation follows the original one described in the paper: encoder and decoder each have 6 identical layers.
Each of the layers are composed of:
- Encoder layers: Multi-head attention, followed by position-wise feed-forward layer (with residual connections)
- Decoder layers: Similar to encoder layers, but with an additional multi-head attention layer that performs attention on the encoder output.

All multi-head attention modules have 16 heads for both encoder and decoder layers. This results in a model that has a total of 210,808,832 parameters. 


## Training

### Loss Function
For both of the models, we use Negative Log-Likelihood loss with label smoothing. 
The models output a probability for each word of the vocabulary for the translated sentence.
From this, we can compute $$NLLLoss(\mathbf{\hat{y}}, \mathbf{y})$$ where $$\mathbf{\hat{y}}$$ is the model output and $\mathbf{y}$ is the target. 


$$ Smooth Loss = -mean (log softmax(\mathbf{\hat{y}})) $$
$$ Loss = confidence * NLLLoss + smoothing * SmoothLoss $$

Where $$confidence = 1 - smoothing$$. The smoothing is set to a value of 0.1 for both tasks.

### Optimization
As we have seen above, both models have a very high number of parameters to train. This can be an issue when using GPUs, as the model needs to fit in memory.
Also, back-propagation requires the memory to be at least twice the size of the model for it to work in memory. For that, instead of using regular precision, we used
mixed-precision training, where most computations are done in `Float16`. We use a centralized version of `Adam`, where gradients are aggregated amongst all workers before 
updating weights.

### Datasets
Both tasks use the same test set, but are trained on slightly different data sets:
- The LSTM is trained on the English to German World Machine Translation 16 (WMT16) dataset, comprising of 3,975,116 translated sentences
- The Transformer is trained on the English to German World Machine Translation 17 (WMT17) dataset, comprising of 4,590,101.
 

More details on both tasks can be found on our [documentation](https://mlbench.readthedocs.io/en/latest/benchmark-tasks.html#task-4-machine-translation).

## Results

Let us now get to fun part; the results. As previously discussed, those models have important training times, and the aim of MLBench is to study the benefit of distribution.
For reproducibility purposes, here is the hardware and software we have used:
- Cloud service: Google Cloud
- Machine Type: `n1-standard-4`
- PyTorch 1.5.1
- NVIDIA Tesla-T4 GPU (1 per node)
- 4 cores and 15GB of RAM
- NCCL communication backend 

The goal for both models is determined by the Bilingual Evaluation Understudy Score (BLEU):
- The LSTM task stops when reaching a BLEU Score of 24.0
- The Transformer task stops when reaching a BLEU score of 25.0

The models are trained on 1,2,4,8 and 16 workers, and all step times are precisely measured to obtain an accurate speed up quantification.
Speedups are computed with respect to the 1 worker case, and are intended to illustrate the distributive capabilities of the task.

### LSTM (GNMT implementation)

The graph below shows the time speedups for the first model. The left graph shows the absolute speed ups with respect to one worker, and the right one omits
communication times from the speed up. This allows us to better see the effect of communication.

<a href="{{ site.baseurl }}public/images/blog/2020-10-02-nlp-translation/task4a_speedup.png" data-lightbox="task4a_speedups" data-title="Speedups for GNMT">
  ![test]({{ site.baseurl }}public/images/blog/2020-10-02-nlp-translation/task4a_speedup.png)
  *GNMT Speedups*
</a>

A few interesting points:
- Overall speedups follows a $$ log_{2}(n) $$, with `n = num_workers`, while compute are roughly linear.
- Scaling the number of compute nodes gives nearly perfect scaling for this task
- Using more powerful communication hardware (e.g. Tesla V100) will positively affect speedups.

The next figure shows the total time spent in each step of training. As expected, we can see that compute steps take less time as we increase the number of nodes,
while communication increasingly takes more and more time, following a logarithmic path. 

Time spent optimizing doesn’t seem to follow the same path, but increases are insignificant (~10 seconds), 
and are due to additional compute steps (averaging tensors, computations related to Mixed precision) when using distribution.

<a href="{{ site.baseurl }}public/images/blog/2020-10-02-nlp-translation/task4a_times.png" data-lightbox="task4a_times" data-title="Step times for GNMT">
  ![test]({{ site.baseurl }}public/images/blog/2020-10-02-nlp-translation/task4a_times.png)
  *Step times for GNMT*
</a>

Finally, the following figure shows the loss evolution (left), Ratio of communication to total time (center), and a price index (right), 
 computed as follows $$ index = \frac{price\_increase}{performance\_increase} $$

<a href="{{ site.baseurl }}public/images/blog/2020-10-02-nlp-translation/task4a_loss_ratio_prices.png" data-lightbox="task4a_loss_ratio_prices" data-title="Step times for GNMT">
  ![test]({{ site.baseurl }}public/images/blog/2020-10-02-nlp-translation/task4a_loss_ratio_prices.png)
  *Step times for GNMT*
</a>

Communication takes up a huge part of training as we increase distribution:  over 70% of the time is spent sending tensors for 16 workers. 
This could be made faster by using a more appropriate connectivity between the workers (currently it is at 10GB/s) that can reduce times by a factor of 10 or more.
An interesting thing to observe is that the curve of cost index first decreases and has a valley before increasing again, which depicts the limits of distribution for this task. 
The price to performance increase seems to be the best for 4 workers, but all indices are lower than 1, meaning the cost compromise is worth it for this task.

### Transformer



-----

## References


{% bibliography --cited %}