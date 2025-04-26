# EE6483 Artificial Intelligence and Data Mining

[中文](./README_CN.md)

# 1.Introduction

This course offers a concise overview of the core theories and techniques in both **Artificial Intelligence** and **Data Mining**, emphasizing **state space representation and search strategies, association rule mining, supervised and unsupervised learning, neural networks, and clustering**. By exploring these methods and their real-world **applications**, students will acquire practical skills to tackle complex problems and uncover valuable insights from data.

This repository serves as a comprehensive resource for students and enthusiasts alike. 

1. **Personal Solutions to Past Exams** – Detailed, step-by-step write-ups of previously tested questions to guide your revision and deepen conceptual understanding.
2. **PPT Example References** – Walkthroughs of example problems and exercises presented in the lecture slides, clarifying key ideas and methodologies.
3. **Analysis of Challenging Topics** – In-depth discussions and breakdowns of complex areas, helping you navigate common pitfalls and master advanced concepts.

If you happen to have a GitHub account and find this repository helpful, **please consider giving it a star⭐**.

![data-sciene-intelligence-artificielle](./README.assets/data-sciene-intelligence-artificielle.png)

# 2.**Course Aims**

This course introduces the fundamental theory and concepts of **Artificial intelligence (AI)** and **Data Mining** methods, in particular **state space representation** and **search strategies, association rule mining, supervised learning, classifiers, neural networks, unsupervised learning, clustering analysis**, and their applications in the area of Artificial Intelligence and Data Mining. This can be summarized as: 

1. To understand the concepts of knowledge representation for state space search, strategies for the search. 
2. To understand the basics of a data mining paradigm known as Association Rule Mining and its application to knowledge discovery problems. 
3. To understand the fundamental theory and concepts of supervised learning, unsupervised learning, neural networks, several learning paradigms and its applications

# 3.Course Content

Structures and Strategies for State Space Representation & Search. 

Heuristic Search. 

Data Mining Concepts and Algorithms. 

Classification and Prediction methods. 

Unsupervised Learning and Clustering Analysis.

# 4.Reading and References

## 4.1Textbooks

1. Luger George F, Artificial Intelligence : Structures and Strategies for Complex Problem Solving, 6 th Edition, Addison-Wesley, 2009. (Q335.L951)

2. Pang-Ning Tan, Michael Steinbach, Vipin Kumar, Introduction to Data Mining: Pearson 2nd Edition, 2019.

3. Ian Goodfellow, Yoshua Bengio and Aaron Courville, Deep Learning, MIT Press, 2016. ISBN: 978-0262035613 (Q325.5.G651)

## 4.2 References

1. Jiawei Han, Micheline Kamber and Jian Pei, Data Mining: Concepts and Techniques, 3rd Edition, Morgan Kaufmann, 2011, ISBN: 978-0-12-381479-1.

2. S. Russell and P. Norvig, Artificial Intelligence -A Modern Approach, 4th Edition, Prentice Hall, 
3. Kevin P. Murphy, Probabilistic Machine Learning- An Introduction, The MIT Press, 2022.
5. Christopher M. Bishop, Pattern Recognition and Machine Learning, Springer, 2006



# 5.Content

## Symbolic Artificial Intelligence and Data Mining

## 0. Weekly Plan

### 1. Introduction to Historical Applications of Artificial Intelligence

1.1 What is Artificial Intelligence?

1.2 Why AI

1.3 Common Goals

1.4 A Brief History

1.5 Applications

1.6 Summary

### 2. Structure and Strategy of Search

#### 2.1 Knowledge Representation

2.1.1 Symbolic Artificial Intelligence

2.1.1.1 8 Difficult Problems

2.1.1.2 Automotive Diagnostic Problems

2.1.1.3 Symbolic Artificial Intelligence

2.1.2 Problem Solving with Search Algorithms

**2.1.3 Knowledge Representations**

2.1.3.1 Example:Farmer Crossing the River

2.1.3.2 Example:8 Puzzles

2.1.3.3 Example:The Seven Bridges of Königsberg Problem 

2.1.3.3.1 State Space Diagrams

2.1.3.3.2 Euler's conclusion

2.1.4 Graphs

2.1.5 Tree

2.1.6 State space search

2.1.6.1 Quaternions [N, A, S, GD]

2.1.6.2 Solution paths

2.1.6.3 Puzzle 8 or 15

2.1.6.4 Exercise 1: Double Kettle

2.1.6.5 Example-Traveler

#### 2.2 Search Strategies

2.2.1 State Space

2.2.1.1 Data-driven search

2.2.1.2 Goal-driven search

2.2.1.3 Similarities and Differences

2.2.1.4 Mixing

2.2.1.5 Example:Descent

2.2.1.6 Summary:L2 & L3

2.2.1.7 Exercise 2: Data or Targets

2.2.2 Graph Search Implementation

2.2.2.1 Why search for diagrams

2.2.2.2 Backtracking Search

2.2.2.2.1 Notation for backtracking

2.2.2.2.2 Backtracking procedure

2.2.2.2.3 Recursion per node

2.2.2.2.4 Main ideas used in backtracking

2.2.2.2.5 Backtracking algorithm

2.2.2.2.6 Backtracking example

2.2.2.2.7 Exercise 3: Backtracking on a Graph

2.2.2.3 BFS breadth-first search

2.2.2.3.1 Implementation of breadth-first search

2.2.2.3.2 Breadth-First Search Algorithm

2.2.2.3.3 Example: Breadth-First Search

2.2.2.3.4 Summary

2.2.2.3.5 Generalized Priority Search: Open and Closed

2.2.2.3.6 Limitations of Breadth-First Search

2.2.2.4 DFS depth-first search

2.2.2.4.1 Depth-first search

2.2.2.4.2 Alternative implementations

2.2.2.4.3 Implementation

2.2.2.4.4 Algorithms

2.2.2.4.5 Example: Depth-first search

2.2.2.4.6 Alternative Implementations

2.2.2.5 Breadth-first vs. depth-first comparison

2.2.2.6 Depth-First Search with Iterative Deepening of DFS-ID

2.2.2.7 Exercise 4: Breadth and Depth

#### 2.3 Summary

### 3. Heuristic Search Games

#### 3.1 Search Strategies

3.1.1 Search strategy without information vs. search strategy with information 

#### 3.2 Introduction:Heuristic search

#### 3.3 Heuristics

3.3.1 Limitations

3.3.2 Two key components

3.3.3 Tic-Tac-Toe Example

#### 3.4 Heuristic Search Algorithms

3.4.1 Mountain Climbing

3.4.1.1 The simplest method

3.4.1.2 Limitations

3.4.1.3 Conventions

3.4.1.4 Examples

3.4.2 Best-first search

3.4.2.1 Algorithm

3.4.2.2 Example

3.4.2.3 Summary

3.4.3.1 Heuristics-Example 1

3.4.3.2 Heuristic Design-Example 2

3.4.4 Design Heuristics

3.4.5 Best First Search

3.4.5.1 Greedy best-first search

3.4.5.2 A-Star Algorithm

3.4.5.3 Example

3.4.5.3.1A*-8 Puzzle

3.4.5.3.2BFS&DFS&A*

3.4.5.3.3 Example:Greedy Paths

3.4.5.3.4 Example:A-star paths

3.4.5.4 Properties of Greedy and A*

3.4.5.5 Exercises

3.4.5.6 Summary

3.4.5.7 Summary

#### 3.5 Heuristic Search and Games

3.5.1 Search and Games

3.5.2 Games and Artificial Intelligence

3.5.2.1 Minimalized Maximal Algorithm 

3.5.2.2 Game Trees

3.5.2.3 Games as Search Problems

3.5.2.4 Example:A Game Tree for Tic-Tac-Toe

**3.5.3 The Minimal-Extremely-Great Algorithm

3.5.3.1 Introduction

3.5.3.2 The NIM game

3.5.3.3 Tic-Tac-Toe

3.5.3.4 Properties

**3.5.3.5 Exercise 4: Very Small Very Large-Tree**

3.5.4 Alpha-Beta Algorithm

3.5.4.1 Beta Pruning

3.5.4.2 alpha pruning

3.5.4.3 Summary

3.5.4.4 Key Ideas

3.5.4.5 Algorithm

3.5.4.6 Example:Extremely Small Extremely Large Decisions

**3.5.4.7 Example:Alpha-Beta Pruning**

3.5.4.8 Exercise 5

3.5.4.9 Exercise 56

3.5.4.10 Properties

3.5.4.11 Monte Carlo Tree Search

3.5.4.12 Summary

### 4. Introduction to Data Mining and Correlation Analysis

#### 4.1 Introduction to Data Mining

4.1.1 What is data?

4.1.1.1 Data Matrix

4.1.1.2 Transaction data

4.1.1.3 Graph data, genome sequence data

4.1.1.4 Data, Information, Knowledge

4.1.1.5 Data, Information, Knowledge, Wisdom

4.1.2 What is data mining?

4.1.2.1 Discuss:Which of the following activities is a data mining task

4.1.2.2 Data Mining in Business Intelligence

4.1.2.3 Knowledge Discovery Processes in Databases: A Typical View from Machine Learning and Statistical Perspectives

4.1.3 Data Mining Tasks

4.1.4 Types of Data Mining = Key Functions

4.1.4.1 What types of data are mined?

4.1.5 Data Mining Applications

4.1.5.1 Architecture: Typical Data Mining Systems

4.1.5.2 Main problems of data mining

4.1.6 Exercise 4.1

#### 4.2 Correlation Analysis

4.2.1 Motivation

4.2.2 What is association rule mining

4.2.2.1 Basic concepts

4.2.2.2 More basic concepts

4.2.2.3 Support and confidence level

4.2.2.4 Association Rule Mining

4.2.2.5 Two tasks

4.2.3 Frequent itemset generation

4.2.4 Apriori Principle

4.2.4.1 Principle

4.2.4.2 Pruning

4.2.4.3 Algorithm

4.2.4.4 Principle

4.2.4.5 Generate candidate set

4.2.4.6 Example 1

4.2.4.7 Example 2

4.2.4.8 Example 3

**4.2.5 The FP-Growth Algorithm**

4.2.5.1 Algorithm

4.2.5.2 FP-Tree

4.2.5.3 Conditional FP-Tree

4.2.5.4 Algorithm

4.2.5.5 Conditional pattern base

4.2.5.6 Principles

4.2.5.7 Conclusion

4.2.5.8 Exercise 4.2

4.2.6 Association Rule Generation

4.2.6.1 Example 1

4.2.6.2 Example 2

4.2.6.3 Pruning

4.2.6.4 Association Rule Generation

4.2.6.5 Exercise 4.3

4.2.7 Association Rule Evaluation

4.2.7.1 Lift

4.2.8 Summary

#### 4.3 Additional Material

4.3.1 Maximum Frequent Item Sets

4.3.2 Closed frequent itemsets

4.3.3 Examples

4.3.4 Summary

4.3.5 ARM-oriented deep learning

4.3.6 Hybrid ARM

4.3.7 Exercise 4.1

4.3.9 Exercise 4.2

4.3.10 Exercise 4.3

## Machine learning

## 0. General Information

### 1. Introduction to Machine Learning

1.1 Introduction to Machine Learning

1.1.1 Artificial Intelligence

1.1.2 Machine Learning

1.1.3 Deep Learning

1.2 Types of Machine Learning

1.2.1 Supervised Learning

1.2.2 Unsupervised learning

1.2.3 Reinforcement Learning

1.3 A Brief History of Machine Learning

1.3.1 Neuronal Models

1.3.2 Perceptrons

1.3.3 Backpropagation

1.3.4 Convolutional Neural Networks

1.3.5 Long-term short-term memory

1.3.6 Support vector machines

1.3.7 lmageNet

1.4 Image datasets

1.5 Applications

1.5.1 Examples of machine learning applications

### 2. Classification and decision trees

2.1 Introduction

2.2 Typical Classification Methods

2.3 Data Samples and Attributes

2.3.1 Types of Attributes

2.3.1.1 Nominal

2.3.1.2 Ordered

2.3.1.3 Interval

2.3.1.4 Ratio

2.4 Classification Models

2.4.1 Decision trees

2.4.2 Random Forest

2.4.2.1 Evaluating Classifier Performance

2.4.2.1.1 Proper nouns

2.4.2.1.2 Examples

2.4.2.1.3 Other aspects

2.4.2.1.4 Data Segmentation

2.5 Summary

### 3. Nearest Neighbor Classifiers and Support Vector Machines

#### 3.1 Nearest Neighbor Classifiers

3.1.1 K Nearest Neighbor Classifier

3.1.2 Dissimilarity and similarity measures

3.1.2.1 Minkowski distance

3.1.2.2 Cosine Similarity

3.1.3 Picture examples

#### 3.2 Support Vector Machines

3.2.1 Hyperplane

3.2.2 Intervals

3.2.3 Linear decision boundary

3.2.4 Normal vectors

3.2.5 Labeling categories

3.2.6 Distance and support vectors

3.2.7 Lagrange multiplier method

3.2.8 Indivisible cases

3.2.9 Multi-category classification

#### 3.3 Summary

#### 3.4 Linear Support Vector Machine Models

#### 3.5 SVM Example

### 4. Neural Networks

#### 4.1 Introduction

4.1.1 Outline

4.1.2 Key Terms

4.1.3 Neural Networks

#### 4.2 Perceptual machines

4.2.1 Inputs and outputs

4.2.2 Models

4.2.3 Examples

4.2.3.1 Linear categorizable classifications

4.2.3.2 Linear Unclassifiable Classification

#### 4.3 Neural Networks

4.3.1 Multilayer perceptron

4.3.2 Feedforward neural network

4.3.3 Examples

4.3.4 Activation functions

4.3.4.1 Linear function

4.3.4.2 Sigma function

4.3.4.3 tanh function

4.3.4.4Sign function

4.3.4.5 ReLu function

4.3.4.6 Leaky ReLu function

4.3.4.7 Demo

4.3.5 Backpropagation

4.3.5.1 Gradient descent

4.3.5.2 Backpropagation algorithm

4.3.5.2.1 Example

#### 4 Convolutional Neural Networks

4.4.1 Fully connected neural networks

4.4.2 Example

4.4.3 Overfitting

4.4.4 Definition

4.4.5 Convolution Demonstration

4.4.6 Convolution and feature maps

4.4.6.1 Example

4.4.7 Weights

4.4.8 Step size

4.4.9 Pooling

4.4.10 Regularization

4.4.11 Convolutional layers

4.4.12 Famous convolutional neural networks

4.4.12.1 LetNet-5

4.4.12.2 ImageNet

4.4.12.3 AlexNet

4.4.12.4 Visualizing Convolutional Networks

4.4.12.5 GoogLeNet

4.4.12.6 ResiduaINet

4.4.12.7 ImageNet top 5

#### 4.5 Recurrent Neural Networks

4.5.1 Example

#### 4.6 Auto-encoder

4.6.1 Application

#### 4.7 Generating Adversarial Networks

4.7.1 Example

4.7.1.1 Cyclically consistent adversarial networks

4.7.1.2 Style generating adversarial network

#### 4.8 Transformers

4.8.1 Encoders

4.8.2 Decoder

#### 4.9 Diffusion Models

#### 4.10 Summary

## Data processing

## 0. Mini-projects

## 1. Clustering and Regression

#### 1.1 Mini-project

#### 1.2 Clustering

1.2.1 Introduction
1.2.1.1 Outline
1.2.1.2 Problem
1.2.1.3 Classification
1.2.1.4 Classification to Clustering
1.2.2 Definitions
1.2.2.1 Unsupervised Learning
1.2.2.2 Clustering
1.2.2.3 Cluster
1.2.2.3 Classification vs. Clustering
1.2.3 Similarity
1.2.3.1 Distance metrics/indicators
1.2.3.1.1 Distance d
1.2.3.1.2 Non-negativity
1.2.3.1.3 Triangular inequalities
1.2.3.1.4 Symmetry
1.2.3.1.5 Examples
1.2.3.1.5.1 Euclidean
1.2.3.1.5.2 Manhattan
1.2.3.1.5.3 Upper bound of infinity
1.2.3.1.5.4 Examples
1.2.4 Algorithms
1.2.4.1 Partitioning
1.2.4.2 Hierarchy
1.2.4.3 K-means
1.2.4.3.1 Example 1
1.2.4.3.2 Example 2
1.2.4.3.3 Example 3
1.2.4.3.4 Cost function
1.2.4.3.5 Advantages and Disadvantages
1.2.4.4 Hierarchical clustering
1.2.4.4.1 Definitions
1.2.4.4.2 Steps
1.2.4.4.3 Visualization
1.2.4.4.4 Strengths
1.2.4.4.5 Distance
1.2.4.4.5.1 Single chain
1.2.4.4.5.2 Full chain
1.2.4.4.5.3 Even chaining
1.2.4.4.5.4 Center of mass
1.2.4.4.6 Distance-minimizing clusters
1.2.4.4.6.1 Steps
1.2.4.4.6.2 Tree diagrams
1.2.4.4.7 Example
1.2.4.5 K-means and Hierarchical Clustering
1.2.5 Summary

#### 1.3 Regression

1.3.1 Introduction
1.3.1.1 Outline
1.3.1.2 Problems
1.3.1.3 Classification
1.3.1.4 Classification to Regression
1.3.2 Linear Regression
1.3.2.1 Definitions
1.3.2.2 Example
1.3.2.3 Derivation
1.3.3 Summary



### 2. Regularization and Optimization

#### 2.1 Review

2.1.1 K-means
2.1.2 HAC
2.1.2.1 Examples
2.1.3 Linear regression

#### 2.2 Introduction

2.2.1 Outline
2.2.2 Problems

#### 2.3 Supervised vs. Unsupervised Learning

2.3.1 Types
2.3.2 Classification
2.3.3 Supervised Learning Framework
2.3.3.1 Mathematical Representation
2.3.4 Learning Effects
2.3.4.1 Overfitting
2.3.4.2 Underfitting
2.3.4.3 Errors
2.3.4.4 Challenges
2.3.4.4.1 Exact Models
2.3.4.4.2 Data Distribution
2.3.4.5 Practice
2.3.4.5.1 Assumptions
2.3.4.5.2 Empirical Risk Minimization
2.3.5 Learning Error
2.3.5.1 Error
2.3.5.1.1 Deviation
2.3.5.1.2 Variance
2.3.5.1.3 Expectation error
2.3.5.2 Model complexity
2.3.5.2.1 Simple-high error
2.3.5.2.2 Complex - high variance
2.3.5.2.3 Balanced

#### 2.4 Statistical Learning (not examined)

2.4.1 Deriving the Expected Error
2.4.2 Statistical Learning

#### 2.5 Overfitting and Underfitting

2.5.1 Good Models
2.5.2 Simple and complex models
2.5.3 Testing for Error
2.5.4 Tradeoffs

#### 2.6 Optimization and Regularized Learning

2.6.1 Validation Sets
2.6.2 Model Training Diagnostics
2.6.2.1 Learning Rate
2.6.2.2 Regularization
2.6.2.2.1 Dropout
2.6.2.2.2 Early Stopping
2.6.2.2.3 Weight Sharing
2.6.2.3 Solutions
2.6.3 Data Enhancement
2.6.3.1 Geometry
2.6.3.2 Photometry
2.6.3.3 Other
2.6.4 Solutions

#### 2.7 Summary



### 3. PCA and Bayesian Inference

#### 3.1 Review

3.1.1 Bias and Variance
3.1.2 Overfitting and Underfitting
3.1.3 Regularization

#### 3.2 Dimensionality Reduction

3.2.1 Outline
3.2.2 Problems

#### 3.3 Unsupervised Learning

3.3.1 Clustering vs. dimensionality reduction
3.3.2 Dimensionality reduction
3.3.3 Objectives
3.3.4 Causes

#### 3.4 Principal Component Analysis

3.4.2 Definitions
3.4.2 Optimization
3.4.3 Steps
3.4.4 Examples
3.4.5 Derivation
3.4.6 Examples

#### 3.5 Summary

#### 3.6 Bayesian Inference

3.6.1 Outline
3.6.2 Problems
3.6.3 Probability
3.6.4 Probability theory
3.6.5 Joint and conditional probability
3.6.6 Bayes' Theorem
3.6.7 Examples
3.6.7.1 Example 1
3.6.7.2 Example 2
3.6.7.3 Example 3
3.6.7.4 Example 4
3.6.8 Plain Bayes
3.6.9 Example
3.6.10 Summary



### 4. Summary

4.1 Types of Supervision
4.1.1 Supervised to Unsupervised
4.2 Clustering
4.2.1 Clustering Algorithms
4.2.2 K-means
4.2.3 HAC
4.2.3.1 HAC example
4.3 Linear Regression
4.4 PCA
4.5 Deviation and Variance
4.6 Overfitting and Underfitting
4.6.1 Examples
4.7 Bayesian Inference
4.7.1 Example 1
4.7.2 Example 2

Translated with DeepL.com (free version)

# 6.List of GitHub

```

```

# 7.Disclaimer

All content in this  is based solely on the contributors' personal work, Internet data.
All tips are for reference only and are not guaranteed to be 100% correct.
If you have any questions, please submit an Issue or PR.
In addition, if it infringes your copyright, please contact us to delete it, thank you.



#### Copyright © School of Electrical & Electronic Engineering, Nanyang Technological University. All rights reserved.
