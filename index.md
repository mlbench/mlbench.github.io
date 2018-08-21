---
layout: default
title: Home
---
<h1>mlbench: Distributed Machine Learning Benchmark</h1>

<a href="https://travis-ci.com/mlbench/mlbench"><img src="https://travis-ci.com/mlbench/mlbench.svg?branch=develop"></a>
<a href="https://mlbench.readthedocs.io/en/latest/?badge=latest"><img src="https://readthedocs.org/projects/mlbench/badge/?version=latest" alt="Documentation Status"></a>





<p>A public and reproducible collection of reference implementations and benchmark suite for distributed machine learning systems. Benchmark for large scale solvers, implemented on different software frameworks & systems.
<strong>This is a work in progress and not usable so far</strong>


    <ul>
        <li>Free software: Apache Software License 2.0</li>
        <li>Documentation: <a href="https://mlbench.readthedocs.io">https://mlbench.readthedocs.io</a>.</li>
    </ul>
</p>

<p>
    <h2>Features</h2>

    <ul>
        <li>For reproducibility and simplicity, we currently focus on standard <strong>supervised ML</strong>, namely classification and regression solvers.</li>
        <li>We provide <strong>reference implementations</strong>for each algorithm, to make it easy to port to a new framework.</li>
        <li>Our goal is to benchmark all/most currently relevant <strong>distributed execution frameworks</strong>. We welcome contributions of new frameworks in the benchmark suite</li>
        <li>We provide <strong>precisely defined tasks</strong> and datasets to have a fair and precise comparison of all algorithms and frameworks.</li>
        <li>Independently of all solver implementations, we provide universal <strong>evaluation code</strong> allowing to compare the result metrics of different solvers and frameworks.</li>
        <li>Our benchmark code is easy to run on the <strong>public cloud</strong>.</li>
    </ul>
</p>

