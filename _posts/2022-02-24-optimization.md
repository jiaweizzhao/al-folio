---
layout: post
title: Optimization in Deep Learning
date: 2022-02-24 21:01:00
description: none
categories: projects
---

Optimization is an important concept in deep learning as the optimization algorithms strongly affect the quality of the networks after training. Non-convex analysis is one of the key issues due to the non-convex landscape of the optimization problem in deep learning. As the learning becomes computationally expensive, there is a growing attention to resource constraint optimization, which aims to reduce the resource consumption during training while preserving the quality of the solutions. In addition, more optimization algorithms should be developed for specific scenarios, such as multi-agent setting. 

In this page, we introduce recent works on optimization published by our group, where the works are categorized by their topics. 


# Learning Algorithms in General Deep Learning

### Madam Optimizer - Multiplicative Weight Update for Deep Learning [(link)](https://arxiv.org/abs/2006.14560)

We propose a multiplicative weight update method Madam - a multiplicative version of the Adam optimizer. We prove that multiplicative weight updates satisfy a descent lemma tailored to compositional functions. Our empirical results show that Madam can train state-of-the-art neural network architectures without learning rate tuning.

### signSGD Optimizer: Sign-based Stochastic Gradient Descent [(link)](https://arxiv.org/abs/1802.04434)

We propose signSGD that updates the weights only using the sign of each minibatch stochastic gradient. Through theoretical analysis, we prove that signSGD matches the SGD-level convergence rate. On the practical side, we find that the momentum counterpart of signSGD is able to match the accuracy and convergence speed of Adam on deep Imagenet models.

# Resource Constraint Optimization

### LNS-Madam: Low-Precision Training in Logarithmic Number System using Multiplicative Weight Update [(link)](https://arxiv.org/abs/2106.13914)

<div class="col-sm mt-3 mt-md-0 mx-auto" style="max-width: 400px;">
    {% include figure.html path="assets/img/add_vs_mul.jpg" class="img-fluid" %}
</div>

We develop a co-designed low-precision training framework LNS-Madam, in which we jointly design the logarithmic number system (LNS) and the multiplicative weight update algorithm Madam. We prove that Madam induces less quantization error as it directly updates the weights in a logarithmic representation. Thus, training with Madam leads to a stable convergence even if precision is strongly limited. 

### signSGD with Majority Vote: Distributed Learning using signSGD Algorithm [(link)](https://arxiv.org/abs/1810.05291)

<div class="col-sm mt-3 mt-md-0 mx-auto" style="max-width: 400px;">
    {% include figure.html path="assets/img/breakdown.jpg" class="img-fluid" %}
</div>

We propose signSGD with majority vote - a robust, communication-efficient learning algorithm for distributed learning. Workers transmit only the sign of their gradient vector to a server, and the overall update is decided by a majority vote. This algorithm uses 32× less communication per iteration than full-precision, distributed SGD. Benchmarking against the state-of-the-art collective communications library (NCCL), our framework leads to a 25% reduction in time for training resnet50 on Imagenet when using 15 AWS p3.2xlarge machines.

# Multi-Agent Optimization

### CGD Optimizer: Competitive Gradient Descent [(link)](https://arxiv.org/abs/1905.12103)

To be filled.

### Competitive Gradient Descent for GAN Training [(link)](https://arxiv.org/abs/1910.05852)

To be filled.

### Polymatrix Competitive Gradient Descent [(link)](https://arxiv.org/abs/2111.08565)

To be filled.

# Policy Optimization

To be filled.

### Competitive Policy Optimization [(link)](https://arxiv.org/abs/2006.10611)

To be filled.

### Deep Bayesian Quadrature Policy Optimization [(link)](https://arxiv.org/abs/2006.15637)

To be filled.

### Trust Region Policy Optimization of POMDPs [(link)](https://authors.library.caltech.edu/94179/1/1810.07900.pdf)

To be filled.

# Non-Convex Optimization Analysis

To be filled.

### Efficient approaches for escaping higher order saddle points in non-convex optimization [(link)](http://arxiv.org/abs/1602.05908)

To be filled.
<!-- ###  [(link)]() -->

