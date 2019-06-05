---
layout: post
title: "Tutorial: Adapting existing PyTorch Code to MLBench"
author: r_grubenmann
published: true
tags: [tutorial,pytorch,guide]
excerpt_separator: <!--more-->
---
In this tutorial, we will go through the process of adapting existing distributed PyTorch code to work with the MLBench framework. This allows you to run your models in the MLBench environment and easily compare them
with our reference implementations as baselines to see how well your code performs.

MLBench is designed to easily be used with third-party models, allowing for quick and fair comparisons and saving all of the hassle that's needed to implement your own baselines for comparison.

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

We also need to change the ``init_processes`` method to reflect our previous changes, along with setting the ``WORLD_SIZE`` and ``RANK`` environment variables:

{% highlight python linenos %}
def init_processes(rank, run_id, hosts, backend='gloo'):
    """ Initialize the distributed environment. """
    hosts = hosts.split(',')
    os.environ['MASTER_ADDR'] = hosts[0] # first worker is the master worker
    os.environ['MASTER_PORT'] = '29500'
    world_size = len(hosts)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['RANK'] = str(rank)
    dist.init_process_group(backend, rank=rank, world_size=len(world_size))
    run(rank, world_size, run_id)
{% endhighlight %}

Next, we need to change the signature of the ``run`` method to add the ``run_id`` parameter. The ``run_id`` is a unique identifier automatically assigned by MLBench to identify an individual run and all its data and performance metrics.

{% highlight python %}
def run(rank, size, run_id):
{% endhighlight %}


At this point, the script could technically already run in MLBench. But so far it would not report back to the Dashboard and you wouldn't be able to see stats during training. So let's add some reporting functionality.

The PyTorch script reports loss to ``stdout``, but we can easily report the loss to MLBench as well. First we need to import the relevant MLBench functionality by adding the following line to the imports at the top of the file:

{% highlight python %}
from mlbench_core.utils import Tracker
from mlbench_core.evaluation.goals import task1_time_to_accuracy_goal
from mlbench_core.evaluation.pytorch.metrics import TopKAccuracy
from mlbench_core.controlflow.pytorch import validation_round
{% endhighlight %}

Then we can simply create a ``Tracker`` object and use it to report the loss and add metrics (``TokKAccuracy``) to track. We add code to record the timing of different steps with ``tracker.record_batch_step()``.
We have to tell the tracker that we're in the training loop ba calling ``tracker.train()`` and that the epoch is done by calling ``tracker.epoch_end()``. The loss is recorded with ``tracker.record_loss()``.

{% highlight python linenos %}
def run(rank, size, run_id):
    """ Distributed Synchronous SGD Example """
    torch.manual_seed(1234)
    train_set, bsz = partition_dataset()
    model = Net()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    metrics = [
        TopKAccuracy(topk=1),
        TopKAccuracy(topk=5)
    ]
    loss_func = nn.NLLLoss()

    tracker = Tracker(metrics, run_id, rank)

    num_batches = ceil(len(train_set.dataset) / float(bsz))

    tracker.start()

    for epoch in range(10):
        tracker.train()

        epoch_loss = 0.0
        for data, target in train_set:
            tracker.batch_start()

            optimizer.zero_grad()
            output = model(data)

            tracker.record_batch_step('forward')

            loss = loss_func(output, target)
            epoch_loss += loss.data.item()

            tracker.record_batch_step('loss')

            loss.backward()

            tracker.record_batch_step('backward')

            average_gradients(model)
            optimizer.step()

            tracker.batch_end()

        tracker.record_loss(epoch_loss, num_batches, log_to_api=True)

        logging.debug('Rank %s, epoch %s: %s',
                      dist.get_rank(), epoch,
                      epoch_loss / num_batches)

        tracker.epoch_end()

        if tracker.goal_reached:
            logging.debug("Goal Reached!")
            return
{% endhighlight %}


That's it. Now the training will report the loss of each worker back to the Dashboard and show it in a nice Graph.

For the official tasks, we also need to report validation stats to the tracker and use the offical validation code. Rename the current ``partition_dataset()`` method to ``partition_dataset_train``
and add a new partition method to load the validation set:

{% highlight python linenos %}
def partition_dataset_val():
    """ Partitioning MNIST validation set"""
    dataset = datasets.MNIST(
        './data',
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307, ), (0.3081, ))
        ]))
    size = dist.get_world_size()
    bsz = int(128 / float(size))
    partition_sizes = [1.0 / size for _ in range(size)]
    partition = DataPartitioner(dataset, partition_sizes)
    partition = partition.use(dist.get_rank())
    val_set = torch.utils.data.DataLoader(
        partition, batch_size=bsz, shuffle=True)
    return val_set, bsz
{% endhighlight %}

Then load the validation set and add the goal for the official task (The Task 1a goal is used for illustration purposes in thsi example):

{% highlight python linenos %}
def run(rank, size, run_id):
    """ Distributed Synchronous SGD Example """
    torch.manual_seed(1234)
    train_set, bsz = partition_dataset_train()
    val_set, bsz_val = partition_dataset_val()
    model = Net()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    metrics = [
        TopKAccuracy(topk=1),
        TopKAccuracy(topk=5)
    ]
    loss_func = nn.NLLLoss()

    goal = task1_time_to_accuracy_goal

    tracker = Tracker(metrics, run_id, rank, goal=goal)

    num_batches = ceil(len(train_set.dataset) / float(bsz))
    num_batches_val = ceil(len(val_set.dataset) / float(bsz_val))

    tracker.start()
{% endhighlight %}

Now all that is needed is to add the validation loop code (``validation_round()``) to run validation in the ``run()`` function. We also check if the goal is reached and stop training if it is.
``validation_round()`` evaluates the metrics on the validation set and reports the results to the Dashboard.

{% highlight python linenos %}
    tracker.record_loss(epoch_loss, num_batches, log_to_api=True)

    logging.debug('Rank %s, epoch %s: %s',
                  dist.get_rank(), epoch,
                  epoch_loss / num_batches)

    validation_round(val_set, model, loss_func, metrics, run_id, rank,
                      'fp32', transform_target_type=None, use_cuda=False,
                      max_batch_per_epoch=num_batches_val, tracker=tracker)

    tracker.epoch_end()

    if tracker.goal_reached:
        logging.debug("Goal Reached!")
        return
{% endhighlight %}

The full code (with some additional improvements) is in our [Github Repo](https://github.com/mlbench/mlbench-benchmarks/blob/master/examples/mlbench-pytorch-tutorial/)

## Creating a Docker Image for Kubernetes

To actually run our code, we need to wrap it in a Docker Image. We could create one from scratch, but it's easier to use the PyTorch Base image provided by MLBench, which already includes everything you might need for executing a PyTorch model.

Create a new file called ``Dockerfile`` in the ``mlbench-pytorch-tutorial`` directory and add the following code:

{% highlight docker linenos %}
FROM mlbench/mlbench-pytorch-base:latest

RUN pip install mlbench-core==1.0.0

# The reference implementation and user defined implementations are placed here.
# ADD ./requirements.txt /requirements.txt
# RUN pip install --no-cache-dir -r /requirements.txt

RUN mkdir /codes
ADD ./train_dist.py /codes/train_dist.py

EXPOSE 29500

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
/conda/bin/python /codes/train_dist.py --hosts {hosts} --rank {rank} --run_id {run_id}
```

The values in brackets will be substituted by MLBench with the correct values and passed to our script.

We also need to set that the command should be executed on all nodes instead of just the Rank 0 Worker.

<a href="{{ site.baseurl }}public/images/Pytorch_New_Run.png" data-lightbox="Pytorch_New_Run" data-title="Create New PyTorch Run">
  <img src="{{ site.baseurl }}public/images/Pytorch_New_Run.png" alt="Create New PyTorch Run" style="max-width:80%;"/>
</a>

Now we're all set to start our experiment. Hit ``Add Run`` and that's it. You just ran a custom model on MLBench.

You should see a graph of the training loss of each worker, along with the combined ``stdout`` and ``stderr`` of all workers.

<a href="{{ site.baseurl }}public/images/pytorch-tutorial-result.png" data-lightbox="Pytorch_Tutorial_Result" data-title="Result of the Tutorial">
  <img src="{{ site.baseurl }}public/images/pytorch-tutorial-result.png" alt="Result of the Tutorial" style="max-width:80%;"/>
</a>

