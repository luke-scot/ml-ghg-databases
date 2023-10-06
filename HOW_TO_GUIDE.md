# Do you have an incomplete GHG emission dataset? You're in the right place!


Available Greenhouse Gas emissions datasets are often incomplete due to inconsistent reporting and poor transparency. Filling the gaps in these datasets allows for more accurate targeting of mitigation strategies
and therefore a faster reduction of overall emissions. 

This page is a guide for practitioners on how to use automated classification methods to complete these gaps. Different problems require different solutions, so this page is an attempt to guide you to the most likely methods that could work for your problem. No guarantees (please don't sue me I'm on an academic salary...), but it works for us!

Once the full paper is published - [click here to view](), and [click here to cite]().

# "How to" guide (under construction)

The figure below provides an outline of the dataset properties that should lead you to a decision about which classifiers are most suitable to your problem. Each of these steps is discussed in the 3 sections below.

![](https://hackmd.io/_uploads/rJfoQPc33.png)


## Step 1 - What type of gap do you have?
Does your dataset resemble dataset 1, 2 or 3 from the figure below?

![](https://hackmd.io/_uploads/r1C0UU6ep.png)


* **Gap level 1** - All entities have known values for at least some time steps. However, some or all entities are missing values at some time steps.
* **Gap level 2** - Some entities have unknown values at all time steps. All of the feature types of the entities without values are shared with at least one of the entities for which some values are known.
* **Gap level 3** - Some entities have unknown values at all time steps. One or more of the feature types for these entities is not shared with any entity for which values are known.
* **Mixture** - If you have a mixture of these scenarios (e.g. missing time steps and missing entities), then the highest level of gap takes precedence.

## Step 2 - How many features do you have?

Simply count how many **independent** features you have in your dataset.

* Example: A dataset with features "GDP, population, GDP per capita" has 2 independent features as the third is simply a combination of the other two.

## Step 3 - Pick your models.
Given your answers to step 1 (level of gap) and step 2 (number of features), follow the decision tree at the top of this "how to" guide to see which type of model is most likely to work for your gap-filling problem. The section below will outline how some of the models work and give some implementation advice for Python. Please feel free to use pieces of code from the repository associated with the paper https://github.com/luke-scot/ml-ghg-databases/tree/main/notebooks/model_run.
### A - Interpolation
Interpolation is the simplest form of gap-filling and simply uses the values on each side of the gap to infill values. For this reason it's use is generally only usful in a gap level 1 problem. For an introduction to interpolation theory see [An Introduction to Numerical Analysis - Suli and Mayers, 2003](https://books.google.co.uk/books?hl=en&lr=&id=hj9weaqJTbQC&oi=fnd&pg=PR7&dq=suli+numerical+analysis&ots=sL2kfJJjwa&sig=PlH14u5PJJjMGMeGvSEvwVx9iI8&redir_esc=y#v=onepage&q=interpolation&f=false).

**Implementation** - 
[pandas "interpolate" function](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.interpolate.html)

### B - Shallow models
"Shallow" learning models are models that are optimised via iterative steps of model update known as epochs. They are termed "shallow" to distinguish them from the more computationally intensive training of "deep" neural networks which will be addressed in section D. The best place to learn about commonly used shallow methods is directly in the [scikit learn documentation](https://scikit-learn.org/stable/supervised_learning.html#supervised-learning) which also provides links to implementation functions. Shallow methods can perform effectively on level 1 and level 2 problems but lack the capacity to learn the complexity required to perform accurately on more difficult level 2 or level 3 problems.

**Implementation** - [scikit-learn supevised learning functions](https://scikit-learn.org/stable/supervised_learning.html#supervised-learning)

Once your code is written for one model you can simply swap the function call for other models so it is sensible to try a few. "Decision trees", "k-nearest neighbours" and "Perceptron" are recommended first functions to try but depending on datasets other models including logistic regression, Stochastic Gradient Descent (SGD), Support Vector Classifier (SVC), passive aggressive classifier, and naive Bayes, may be effective.


### C - Ensemble models

Ensemble models use the output of multiple shallow models to improve overall performance. See [Ensemble learning: A survey - Sagi and Rokach, 2018](https://wires.onlinelibrary.wiley.com/doi/full/10.1002/widm.1249?casa_token=VJ2KooATzTEAAAAA%3AAIwYObLY3Vt-Y5f2c4Y-LNN-cb37qYupx7lVhqwBJcG16bM8E02F2twX9MU9cFov0GEgbyHmVC8ddcc) for a theoretical overview on why this is useful. The added learning ability of ensemble models relative to shallow models allow them to have consistent performance in more complex problems. The advantage over deeper models is the ease of implementation of ensemble models, however deeper models may be necessary to learn particularly complex relationships when features are missing such as for level 3 problems.

**Implementation** - [scikit-learn ensemble learning functions](https://scikit-learn.org/stable/modules/ensemble.html)

Implementation of these models can be done with the same code as your shallow learning models and simply swapping the scikit learn function. The most effective methods for gap-filling classification tend to be "Randomforest", which is a combination of many decision trees, and "AdaBoost", but feel free to try others!

### D - Deep models
"Deep" learning models are models based on neural networks that iteratively learn to perform classification on a task following a specified optimisation regime. THE reference textbook for deep learning is [Deep Learning - Goodfellow et al., 2016](https://books.google.co.uk/books?hl=en&lr=&id=omivDQAAQBAJ&oi=fnd&pg=PR5&dq=goodfellow+2016&ots=MNV4ftqDUX&sig=Cc3qA8aEOdMGd2XnHMOg3WVYEEQ&redir_esc=y#v=onepage&q=goodfellow%202016&f=false) and is a good introduction to the topic. Deep learning models require large datasets and sufficient features to learn from but can perform very well on complex tasks when this is the case. Two disadvantages elative to simpler models are difficulty in implementation and computational resources required. Therefore, simpler models should always be considered first, and if these are ineffective an evaluation should be made to the potential benefits of using deep models before diving in. 

**Implementation** - [pytorch neural network library](https://pytorch.org/docs/stable/index.html)

One can implement an infinite number of different neural network structures. If unfamiliar with implementation we recommended trying the [PyTorch tutorials](https://pytorch.org/tutorials/) or trying to reuse the code from [this paper's Github repository](https://github.com/luke-scot/ml-ghg-databases/blob/main/notebooks/model_run/run_deepModels.ipynb). Here we will briefly mention some of the main types:
* Fully connected - The simplest form of neural network which has a basic "Multilayer Perceptron" version implementable in scikit-learn [[view here]](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier). Deeper versions can be constructed with [Linear neural network layers](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html).
* Recurrent neural networks - These are typically used for time-series problems. The most widely used form is known as an LSTM (Long-Short Term Memory) model and can be implemented with [LSTM layers](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html).
* Convolutional neural networks - These are typically used for image-based classification tasks but can be used in other cases. One widely used implementation is ResNet which uses residual blocks with [convolutional layers](https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html), to improve learning.

### Graph models
Graph representation learning models can be particularly useful for geographically distributed data due to the ease of conversion to datasets intoa graph structure. THE textbook for graph learning is [Graph Representation Learning - Hamilton, 2020](https://books.google.co.uk/books?hl=en&lr=&id=Csj-DwAAQBAJ&oi=fnd&pg=PP2&dq=hamilton+graph+learning&ots=_djxjdgxJP&sig=nq8-TriLUck9sDbza1QqiJIoLa0&redir_esc=y#v=onepage&q=hamilton%20graph%20learning&f=false) and is an excellent introduction to the topic. Two advantages of graph models over other deep learning models are: the ability to extract information from relationships between entities and not just the entities' properties themselves, and the ability to update these models easily with the incorporation of new data. Once again the implementation of these models is more time consuming and computationally expensive than shallow models so shallow models should always be considered first. 

**Implementation** - [pytorch-geometric library](https://pytorch-geometric.readthedocs.io/en/latest/)

If unfamiliar with implementation we recommended trying the [PyTorch tutorials](https://pytorch-geometric.readthedocs.io/en/latest/notes/colabs.html) or trying to reuse the code from [this paper's Github repository](https://github.com/luke-scot/ml-ghg-databases/blob/main/notebooks/model_run/run_graph.ipynb). Two widely used types of graph model are the [Graph Convolutional Network (GCN)](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.models.GCN.html) and [GraphSAGE](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.models.GraphSAGE.html). Once you have written code to implement one of these models it is straightforward to switch out for the other one so it is worth trying both and maybe others, although beware of computation time.

## Step 4 - Hyperparameter tuning
Once you have established one or a few models that seem to work well for your problem it is worth spending a little more time performing "hyperparameter tuning". This involves adjusting the model's properties, i.e. the options that you can enter into the model implementation functions, to optimise performance. You can do this manually by simply trying a few different values, or comprehensively by running a loop to try all different combinations. This latter may be computationally expensive and unnecessary, other techniques for exploring hyperparameters combinations are explained in the textbook [Hyperparameter Tuning with Python - Owen, 2022](https://books.google.co.uk/books?hl=en&lr=&id=CqF-EAAAQBAJ&oi=fnd&pg=PP1&dq=hyperparameter+tuning+in+python&ots=ROPBsnH0qD&sig=K_MftSABaSdHtET-lKxl3zexbEo&redir_esc=y#v=onepage&q=hyperparameter%20tuning%20in%20python&f=false).