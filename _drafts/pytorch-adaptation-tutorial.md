---
layout: post
title: "Tutorial: Adapting existing PyTorch code to MLBench"
author: r_grubenmann
published: true
tags: [tutorial,pytorch,guide]
excerpt_separator: <!--more-->
---
In this tutorial, we will go through the process of adapting existing distributed PyTorch code to work with the MLBench framework.

MLBench is designed to easily be used with third-party models, allowing for quick and fair comparisons with our standard benchmarks, saving all of the hassle that's needed to implement your own baselines for comparison.

<!--more-->

We will adapt the code from the official [PyTorch distributed tutorial](https://pytorch.org/tutorials/intermediate/dist_tuto.html) to run in MLBench. If you're unfamiliar with that tutorial, it might be worth giving it a quick look so you know what' we're working with.


## Adapting the Code

To get started, create a new directory ``mlbench-pytorch-tutorial`` and copy the [train_dist.py](https://github.com/seba-1511/dist_tuto.pth/blob/gh-pages/train_dist.py) file into it.

The official tutorial spawns multiple parallel processes on a single machine, but we want to run the code on multiple machines, so first we need to replace the initialization functionality with our own.

Replace

{% highlight python linenos %}
if __name__ == "__main__":
    size = 2
    processes = []
    for rank in range(size):
        p = Process(target=init_processes, args=(rank, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
{% endhighlight %}

with

{% highlight python linenos %}
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process run parameters')
    parser.add_argument('--run_id', type=str, help='The id of the run')
    parser.add_argument('--rank', type=int, help='The rank of this worker')
    parser.add_argument('--hosts', type=str, help='The list of hosts')
    args = parser.parse_args()
    init_processes(args.rank, args.run_id, args.hosts)
{% endhighlight %}

and add

{% highlight python %}
import argparse
{% endhighlight %}

to the top of the file.

We also need to change the ``init_processes`` method to reflect our previous changes:

{% highlight python linenos %}
def init_processes(rank, run_id, hosts, backend='gloo'):
    """ Initialize the distributed environment. """
    hosts = hosts.split(',')
    os.environ['MASTER_ADDR'] = hosts[0] # first worker is the master worker
    os.environ['MASTER_PORT'] = '29500'
    world_size = len(hosts)
    dist.init_process_group(backend, rank=rank, world_size=len(world_size))
    run(rank, world_size, run_id)
{% endhighlight %}

We also need to change the signature of the ``run`` method to add the ``run_id`` parameter. The ``run_id`` is a unique identifier automatically assigned by MLBench to identify an individual run and all its data and performance metrics.

{% highlight python %}
def run(rank, size, run_id):
{% endhighlight %}


At this point, the script could technically already run in MLBench. But so far it would not report back to the Dashboard and you wouldn't be able to see stats during training. So let's add some reporting functionality.

The PyTorch script reports loss to ``stdout``, but we can easily report the loss to MLBench as well. First we need to import the relevant MLBench functionality by adding the following line to the imports at the top of the file:

{% highlight python %}
from mlbench_core.api import ApiClient
{% endhighlight %}

Then we can simply create an api client object and use it to report the loss. We instantiate the client as shown on lines 10 - 13 in this snippet and post the loss as shown on lines 32 - 35:

{% highlight python linenos %}
def run(rank, size, run_id):
    """ Distributed Synchronous SGD Example """
    torch.manual_seed(1234)
    train_set, bsz = partition_dataset()
    model = Net()
    model = model
    # model = model.cuda(rank)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    api_client = ApiClient(
        in_cluster=True,
        k8s_namespace='default',
        label_selector='component=master,app=mlbench')

    num_batches = ceil(len(train_set.dataset) / float(bsz))
    for epoch in range(10):
        epoch_loss = 0.0
        for data, target in train_set:
            data, target = Variable(data), Variable(target)
            # data, target = Variable(data.cuda(rank)), Variable(target.cuda(rank))
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            epoch_loss += loss.data[0]
            loss.backward()
            average_gradients(model)
            optimizer.step()
        print('Rank ',
              dist.get_rank(), ', epoch ', epoch, ': ',
              epoch_loss / num_batches)

        api_client.post_metric(
            run_id,
            "Rank {} loss".format(rank),
            epoch_loss / num_batches)
{% endhighlight %}

Make sure to change ``default`` on line 12 to the namespace MLBench is running under in Kubernetes.

That's it. Now the training will report the loss of each worker back to the Dashboard and show it in a nice Graph.

## Creating a Docker Image for Kubernetes

To actually run our code, we need to wrap it in a Docker Image. We could create one from scratch, but it's easier to use the PyTorch Base image provided by MLBench, which already includes everything you might need for executing a PyTorch model.

Create a new file called ``Dockerfile`` in the ``mlbench-pytorch-tutorial`` directory and add the following code:

{% highlight docker linenos %}
FROM mlbench-pytorch-base:latest

RUN pip install mlbench-core

# The reference implementation and user defined implementations are placed here.
# ADD ./requirements.txt /requirements.txt
# RUN pip install --no-cache-dir -r /requirements.txt

RUN mkdir /codes
ADD ./train_dist.py /codes/train_dist.py

ENV PYTHONPATH /codes
{% endhighlight %}

The ``mlbench-pytorch-base:latest`` image already contains all neccessary libraries, but if your image requires additional python libraries, you can add them with the commands on lines 6 and 7, along with adding a ``requirements.txt`` file.

In order for Kubernetes to access the image, you have to build and upload it to a Docker registry that's accessible to Kubernetes, for instance [Docker Hub](https://hub.docker.com/) (Make sure to change the Docker image and repo name accordingly):

```shell
$ docker login
$ docker build -t mlbench/pytorch-tutorial:latest mlbench-pytorch-tutorial/
$ docker push mlbench/pytorch-tutorial:latest
```

The image is now built and available fur running in MLBench

## Running the code in MLBench

Navigate to the MLBench Dashboard and go to the ``Runs`` page.

Create a new Run:

<a href="{{ site.baseurl }}public/images/New_Run.png" data-lightbox="New_Run" data-title="New Run Page">
  <img src="{{ site.baseurl }}public/images/New_Run.png" alt="New Run Page" style="max-width:80%;"/>
</a>

Enter the URL of the newly uploaded Docker image (The host can be left out if you use Docker Hub). Then enter the command to execute on each worker:

```shell
/conda/bin/python /codes/train_dist.py --hosts {hosts} --rank {rank} --run_id = {run_id}
```

The values in brackets will be substituted by MLBench with the correct values and passed to our script.

We also need to set that the command should be executed on all nodes instead of just the Rank 0 Worker.

<a href="{{ site.baseurl }}public/images/Pytorch_New_Run.png" data-lightbox="Pytorch_New_Run" data-title="Create New PyTorch Run">
  <img src="{{ site.baseurl }}public/images/Pytorch_New_Run.png" alt="Create New PyTorch Run" style="max-width:80%;"/>
</a>

Now we're all set to start our experiment. Hit ``Add Run`` and that's it. You just ran a custom model on MLBench.

You should see a graph of the training loss of each worker, along with the combined ``stdout`` and ``stderr`` of all workers.
