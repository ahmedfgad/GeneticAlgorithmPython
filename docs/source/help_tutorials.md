# Tutorials and Resources

Tutorials, articles, and a book about PyGAD and the genetic algorithm.

## Tutorials about PyGAD

### [Adaptive Mutation in Genetic Algorithm with Python Examples](https://neptune.ai/blog/adaptive-mutation-in-genetic-algorithm-with-python-examples)

In this tutorial, we’ll see why mutation with a fixed number of genes is bad, and how to replace it with adaptive mutation. Using the [PyGAD Python 3 library](https://pygad.readthedocs.io/), we’ll discuss a few examples that use both random and adaptive mutation.

### [Clustering Using the Genetic Algorithm in Python](https://blog.paperspace.com/clustering-using-the-genetic-algorithm)

This tutorial discusses how the genetic algorithm is used to cluster data, starting from random clusters and running until the optimal clusters are found. We'll start by briefly revising the K-means clustering algorithm to point out its weak points, which are later solved by the genetic algorithm. The code examples in this tutorial are implemented in Python using the [PyGAD library](https://pygad.readthedocs.io/).

### [Working with Different Genetic Algorithm Representations in Python](https://blog.paperspace.com/working-with-different-genetic-algorithm-representations-python)

Depending on the nature of the problem being optimized, the genetic algorithm (GA) supports two different gene representations: binary, and decimal. The binary GA has only two values for its genes, which are 0 and 1. This is easier to manage as its gene values are limited compared to the decimal GA, for which we can use different formats like float or integer, and limited or unlimited ranges.

This tutorial discusses how the [PyGAD](https://pygad.readthedocs.io/) library supports the two GA representations, binary and decimal.

### [5 Genetic Algorithm Applications Using PyGAD](https://blog.paperspace.com/genetic-algorithm-applications-using-pygad)

This tutorial introduces PyGAD, an open-source Python library for implementing the genetic algorithm and training machine learning algorithms. PyGAD supports 19 parameters for customizing the genetic algorithm for various applications.

Within this tutorial we'll discuss 5 different applications of the genetic algorithm and build them using PyGAD.

### [Train Neural Networks Using a Genetic Algorithm in Python with PyGAD](https://heartbeat.fritz.ai/train-neural-networks-using-a-genetic-algorithm-in-python-with-pygad-862905048429?gi=ba58ee6b4bbd)

The genetic algorithm (GA) is a biologically-inspired optimization algorithm. It has in recent years gained importance, as it’s simple while also solving complex problems like travel route optimization, training machine learning algorithms, working with single and multi-objective problems, game playing, and more.

Deep neural networks are inspired by the idea of how the biological brain works. It’s a universal function approximator, which is capable of simulating any function, and is now used to solve the most complex problems in machine learning. What’s more, they’re able to work with all types of data (images, audio, video, and text).

Both genetic algorithms (GAs) and neural networks (NNs) are similar, as both are biologically-inspired techniques. This similarity motivates us to create a hybrid of both to see whether a GA can train NNs with high accuracy.

This tutorial uses [PyGAD](https://pygad.readthedocs.io/), a Python library that supports building and training NNs using a GA. [PyGAD](https://pygad.readthedocs.io/) offers both classification and regression NNs.

### [Building a Game-Playing Agent for CoinTex Using the Genetic Algorithm](https://blog.paperspace.com/building-agent-for-cointex-using-genetic-algorithm)

In this tutorial we'll see how to build a game-playing agent using only the genetic algorithm to play a game called [CoinTex](https://play.google.com/store/apps/details?id=coin.tex.cointexreactfast&hl=en), which is developed in the Kivy Python framework. The objective of CoinTex is to collect the randomly distributed coins while avoiding collision with fire and monsters (that move randomly). The source code of CoinTex can be found [on GitHub](https://github.com/ahmedfgad/CoinTex).

The genetic algorithm is the only AI used here; there is no other machine/deep learning model used with it. We'll implement the genetic algorithm using [PyGad](https://blog.paperspace.com/genetic-algorithm-applications-using-pygad/). This tutorial starts with a quick overview of CoinTex followed by a brief explanation of the genetic algorithm, and how it can be used to create the playing agent. Finally, we'll see how to implement these ideas in Python.

The source code of the genetic algorithm agent is available [here](https://github.com/ahmedfgad/CoinTex/tree/master/PlayerGA), and you can download the code used in this tutorial from [here](https://github.com/ahmedfgad/CoinTex/tree/master/PlayerGA/TutorialProject).

### [How To Train Keras Models Using the Genetic Algorithm with PyGAD](https://blog.paperspace.com/train-keras-models-using-genetic-algorithm-with-pygad)

PyGAD is an open-source Python library for building the genetic algorithm and training machine learning algorithms. It offers a wide range of parameters to customize the genetic algorithm to work with different types of problems.

PyGAD has its own modules that support building and training neural networks (NNs) and convolutional neural networks (CNNs). Despite these modules working well, they are implemented in Python without any additional optimization measures. This leads to comparatively high computational times for even simple problems.

The latest PyGAD version, 2.8.0 (released on 20 September 2020), supports a new module to train Keras models. Even though Keras is built in Python, it's fast. The reason is that Keras uses TensorFlow as a backend, and TensorFlow is highly optimized.

This tutorial discusses how to train Keras models using PyGAD. The discussion includes building Keras models using either the Sequential Model or the Functional API, building an initial population of Keras model parameters, creating an appropriate fitness function, and more.

[![PyGAD+Keras](https://user-images.githubusercontent.com/16560492/111009628-2b372500-8362-11eb-90cf-01b47d831624.png)](https://blog.paperspace.com/train-keras-models-using-genetic-algorithm-with-pygad)

### [Train PyTorch Models Using Genetic Algorithm with PyGAD](https://neptune.ai/blog/train-pytorch-models-using-genetic-algorithm-with-pygad)

[PyGAD](https://pygad.readthedocs.io/) is a genetic algorithm Python 3 library for solving optimization problems. One of these problems is training machine learning algorithms.

PyGAD has a module called [pygad.kerasga](https://github.com/ahmedfgad/KerasGA). It trains Keras models using the genetic algorithm. On January 3rd, 2021, a new release of [PyGAD 2.10.0](https://pygad.readthedocs.io/) brought a new module called [pygad.torchga](https://github.com/ahmedfgad/TorchGA) to train PyTorch models. It’s very easy to use, but there are a few tricky steps.

So, in this tutorial, we’ll explore how to use PyGAD to train PyTorch models.

[![PyGAD+PyTorch](https://user-images.githubusercontent.com/16560492/111009678-5457b580-8362-11eb-899a-39e2f96984df.png)](https://neptune.ai/blog/train-pytorch-models-using-genetic-algorithm-with-pygad)

### [A Guide to Genetic ‘Learning’ Algorithms for Optimization](https://towardsdatascience.com/a-guide-to-genetic-learning-algorithms-for-optimization-e1067cdc77e7)

## For More Information

There are different resources that can be used to get started with the genetic algorithm and building it in Python. 

### Tutorial: Implementing Genetic Algorithm in Python

To start with coding the genetic algorithm, you can check the tutorial titled [**Genetic Algorithm Implementation in Python**](https://www.linkedin.com/pulse/genetic-algorithm-implementation-python-ahmed-gad) available at these links:

- [LinkedIn](https://www.linkedin.com/pulse/genetic-algorithm-implementation-python-ahmed-gad)
- [Towards Data Science](https://towardsdatascience.com/genetic-algorithm-implementation-in-python-5ab67bb124a6)
- [KDnuggets](https://www.kdnuggets.com/2018/07/genetic-algorithm-implementation-python.html)

[This tutorial](https://www.linkedin.com/pulse/genetic-algorithm-implementation-python-ahmed-gad) is prepared based on a previous version of the project but it still a good resource to start with coding the genetic algorithm.

[![Genetic Algorithm Implementation in Python](https://user-images.githubusercontent.com/16560492/78830052-a3c19300-79e7-11ea-8b9b-4b343ea4049c.png)](https://www.linkedin.com/pulse/genetic-algorithm-implementation-python-ahmed-gad)

### Tutorial: Introduction to Genetic Algorithm

Get started with the genetic algorithm by reading the tutorial titled [**Introduction to Optimization with Genetic Algorithm**](https://www.linkedin.com/pulse/introduction-optimization-genetic-algorithm-ahmed-gad) which is available at these links:

* [LinkedIn](https://www.linkedin.com/pulse/introduction-optimization-genetic-algorithm-ahmed-gad)
* [Towards Data Science](https://www.kdnuggets.com/2018/03/introduction-optimization-with-genetic-algorithm.html)
* [KDnuggets](https://towardsdatascience.com/introduction-to-optimization-with-genetic-algorithm-2f5001d9964b)

[![Introduction to Genetic Algorithm](https://user-images.githubusercontent.com/16560492/82078259-26252d00-96e1-11ea-9a02-52a99e1054b9.jpg)](https://www.linkedin.com/pulse/introduction-optimization-genetic-algorithm-ahmed-gad)

### Tutorial: Build Neural Networks in Python

Read about building neural networks in Python through the tutorial titled [**Artificial Neural Network Implementation using NumPy and Classification of the Fruits360 Image Dataset**](https://www.linkedin.com/pulse/artificial-neural-network-implementation-using-numpy-fruits360-gad) available at these links:

* [LinkedIn](https://www.linkedin.com/pulse/artificial-neural-network-implementation-using-numpy-fruits360-gad)
* [Towards Data Science](https://towardsdatascience.com/artificial-neural-network-implementation-using-numpy-and-classification-of-the-fruits360-image-3c56affa4491)
* [KDnuggets](https://www.kdnuggets.com/2019/02/artificial-neural-network-implementation-using-numpy-and-image-classification.html)

[![Building Neural Networks Python](https://user-images.githubusercontent.com/16560492/82078281-30472b80-96e1-11ea-8017-6a1f4383d602.jpg)](https://www.linkedin.com/pulse/artificial-neural-network-implementation-using-numpy-fruits360-gad)

### Tutorial: Optimize Neural Networks with Genetic Algorithm

Read about training neural networks using the genetic algorithm through the tutorial titled [**Artificial Neural Networks Optimization using Genetic Algorithm with Python**](https://www.linkedin.com/pulse/artificial-neural-networks-optimization-using-genetic-ahmed-gad) available at these links:

- [LinkedIn](https://www.linkedin.com/pulse/artificial-neural-networks-optimization-using-genetic-ahmed-gad)
- [Towards Data Science](https://towardsdatascience.com/artificial-neural-networks-optimization-using-genetic-algorithm-with-python-1fe8ed17733e)
- [KDnuggets](https://www.kdnuggets.com/2019/03/artificial-neural-networks-optimization-genetic-algorithm-python.html)

[![Training Neural Networks using Genetic Algorithm Python](https://user-images.githubusercontent.com/16560492/82078300-376e3980-96e1-11ea-821c-aa6b8ceb44d4.jpg)](https://www.linkedin.com/pulse/artificial-neural-networks-optimization-using-genetic-ahmed-gad)

### Tutorial: Building CNN in Python

To start with coding the genetic algorithm, you can check the tutorial titled [**Building Convolutional Neural Network using NumPy from Scratch**](https://www.linkedin.com/pulse/building-convolutional-neural-network-using-numpy-from-ahmed-gad) available at these links:

- [LinkedIn](https://www.linkedin.com/pulse/building-convolutional-neural-network-using-numpy-from-ahmed-gad)
- [Towards Data Science](https://towardsdatascience.com/building-convolutional-neural-network-using-numpy-from-scratch-b30aac50e50a)
- [KDnuggets](https://www.kdnuggets.com/2018/04/building-convolutional-neural-network-numpy-scratch.html)
- [Chinese Translation](http://m.aliyun.com/yunqi/articles/585741)

[This tutorial](https://www.linkedin.com/pulse/building-convolutional-neural-network-using-numpy-from-ahmed-gad)) is prepared based on a previous version of the project but it still a good resource to start with coding CNNs.

[![Building CNN in Python](https://user-images.githubusercontent.com/16560492/82431022-6c3a1200-9a8e-11ea-8f1b-b055196d76e3.png)](https://www.linkedin.com/pulse/building-convolutional-neural-network-using-numpy-from-ahmed-gad)

### Tutorial: Derivation of CNN from FCNN

Get started with the genetic algorithm by reading the tutorial titled [**Derivation of Convolutional Neural Network from Fully Connected Network Step-By-Step**](https://www.linkedin.com/pulse/derivation-convolutional-neural-network-from-fully-connected-gad) which is available at these links:

* [LinkedIn](https://www.linkedin.com/pulse/derivation-convolutional-neural-network-from-fully-connected-gad)
* [Towards Data Science](https://towardsdatascience.com/derivation-of-convolutional-neural-network-from-fully-connected-network-step-by-step-b42ebafa5275)
* [KDnuggets](https://www.kdnuggets.com/2018/04/derivation-convolutional-neural-network-fully-connected-step-by-step.html)

[![Derivation of CNN from FCNN](https://user-images.githubusercontent.com/16560492/82431369-db176b00-9a8e-11ea-99bd-e845192873fc.png)](https://www.linkedin.com/pulse/derivation-convolutional-neural-network-from-fully-connected-gad)

### Book: Practical Computer Vision Applications Using Deep Learning with CNNs

You can also check my book cited as [**Ahmed Fawzy Gad 'Practical Computer Vision Applications Using Deep Learning with CNNs'. Dec. 2018, Apress, 978-1-4842-4167-7**](https://www.amazon.com/Practical-Computer-Vision-Applications-Learning/dp/1484241665) which discusses neural networks, convolutional neural networks, deep learning, genetic algorithm, and more.

Find the book at these links:

- [Amazon](https://www.amazon.com/Practical-Computer-Vision-Applications-Learning/dp/1484241665)
- [Springer](https://link.springer.com/book/10.1007/978-1-4842-4167-7)
- [Apress](https://www.apress.com/gp/book/9781484241660)
- [O'Reilly](https://www.oreilly.com/library/view/practical-computer-vision/9781484241677)
- [Google Books](https://books.google.com.eg/books?id=xLd9DwAAQBAJ)

![Fig04](https://user-images.githubusercontent.com/16560492/78830077-ae7c2800-79e7-11ea-980b-53b6bd879eeb.jpg)

## More Links

https://rodriguezanton.com/identifying-contact-states-for-2d-objects-using-pygad-and/

https://torvaney.github.io/projects/t9-optimised
