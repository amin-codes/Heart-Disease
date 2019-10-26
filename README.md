# :heart: Helper
*Made by Amin B. Z.*

*Note: Only longBeachVA.csv is used for training this model*

This project is an attempt at making a machine-learning model that can predict if a person has heart disease without visiting a doctor with the following inputs:
* age (integer)
* sex (1 = male; 0 = female)
* chest pain location (1 = substernal; 0 = otherwise) 
* pain is provoked by exertion? (1 = provoked by exertion; 0 = otherwise)
* relieved after rest? (1 = relieved after rest; 0 = otherwise) 
* resting blood pressure (in mm Hg on admission to the hospital) (systolic) (e.g., if your blood pressure is 120/80, then enter 120)
* do you smoke? (1 = yes; 0 = no)
* has there been anyone in your family with heart disease? (1 = yes; 0 = no)
* maximum heart rate achieved

# Results from 10 consecutive tests
**Note: Tables are from same tests, they are just put in different tables. Also, there was no 'random-state' value set for these tests.**

*Data Recorded On: April 7, 2018 (Outdated!)*

Predicting 0 (no heart disease):

| Test ID | Precision | Recall | F1-Score | Support |
| :---    | :---      | :---   | :---     | :---    |
|1|0.79|0.71|0.75|21|
|2|0.68|0.94|0.79|16|
|3|0.77|0.77|0.77|22|
|4|0.83|0.86|0.84|22|
|5|0.78|0.82|0.80|22|
|6|0.83|0.75|0.79|20|
|7|0.81|0.91|0.86|23|
|8|0.77|0.81|0.79|21|
|9|0.86|0.90|0.88|20|
|10|0.89|0.76|0.82|21|

Predicting 1 (heart disease):

| Test ID | Precision | Recall | F1-Score | Support |
| :---    | :---      | :---   | :---     | :---    |
|1|0.65|0.73|0.69|15|
|2|0.93|0.67|0.78|21|
|3|0.71|0.71|0.71|17|
|4|0.73|0.67|0.70|12|
|5|0.69|0.64|0.67|14|
|6|0.71|0.80|0.75|15|
|7|0.78|0.58|0.67|12|
|8|0.71|0.67|0.69|15|
|9|0.85|0.79|0.81|14|
|10|0.75|0.88|0.81|17|

[Link to table for averages from both tables](https://docs.google.com/document/d/1yBwZJ6u_dDgA1cqRK91_6qKzs4riiZbD3HULjpo708k/edit?usp=sharing)
# About the Data and Model

The data that I collected from the UC Irvine Machine Learning Repository was somewhat incomplete. In an attempt to fix this issue, I used an [imputer](https://scikit-learn.org/stable/modules/impute.html) to complete the data by inferring from the already collected data, which is sometimes better than outright throwing the incomplete data away.

Also, the data had initially classified the types of heart disease on a scale of 0-4. 0 means heart disease is absent in the person. 1-4 means that there is some form of heart disease present in the person. I replaced values 1-4 with just 1 because my goal was not to distinguish between the different forms. I only wanted to predict if people had heart disease.

The model's core is comprised of a [pipeline](http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) that includes [Kernel Principal component analysis (KPCA)](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.KernelPCA.html) followed by a [Multilayer Perceptron Classifier (MLP)](http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html).

The KPCA is used as a transformer in the pipeline. It essentially helps cluster, or *transform*, the different components, which are the different questions asked, of the training data in a nonlinear fashion. This is useful because otherwise, the data would be scattered in a nonmeaningful way, which would lower the accuracy of the model's predictions.

The MLP is used as a final estimator in the pipeline. It applies different weights to the different questions (let's refer to the questions as neurons), which means that some questions may be relied on more heavily as an indicator for heart disease. The MLP adjusts these weights based on the training data. One row of data from the training data is inputted to the MLP at a time, and the MLP compares its output with the expected output and calculates an error value, which is called stochastic gradient descent. This error value is then sent back, one neuron at a time, so that each neuron adjusts its weights as needed to minimize error, which is called backpropagation. The MLP repeats this learning process with each row of training data.

The pipeline uses the KPCA and MLP, in that order, to maximize the accuracy of the machine-learning model's predictions.
# Requirements to Run Program
* Install Python (version 2.7.14)
* Install packages: numpy (version >= 1.14.2), pandas (version >= 0.21), scikit-learn (version >= 0.19.1), imbalanced-learn (version >= 0.3.3)
# Research
The main issue is that some of the tests are expensive and some of the cheap tests are not very accurate. EKGs (electrocardiogram) cost approximately $50, exercise stress tests cost $175+, and imaging tests cost between $500 and $2,000.

I wanted to create a machine learning model that could accurately predict if one has heart disease via, what I call, “absolute” inputs (inputs that have no uncertainty behind them).

While searching the *University of California, Irvine Machine Learning Repository*, I found the Heart Disease Data Set that is used in training and testing the model.

Approximately 610,000 people die of some type of heart disease every year in the US, and 47% of sudden cardiac deaths do not even occur in the hospital. The large proportion of sudden deaths occurring outside of hospitals means that people are not aware of their heart condition. People probably do not know about their heart condition because the cost of testing is too high and other low-cost tests could be misleading.
# Testing on Myself and a Friend
If the model outputs "0," then that means the user does not have heart disease, but if it outputs "1," then the user has heart disease.

The array respectively represents the input of the user to the following: age, gender, pain location, pain exertion, relieved after rest, resting blood pressure, do they smoke?, do they have a family history of heart disease?, maximum systolic heart rate achieved.

**These tests were performed April, 2018**
![My heart disease predictions](master/My Test.png)
![My friend's heart disease predictions](master/Friend Test.png)
# References
*1. Hungarian Institute of Cardiology. Budapest: Andras Janosi, M.D.*

*2. University Hospital, Zurich, Switzerland: William Steinbrunn, M.D.*

*3. University Hospital, Basel, Switzerland: Matthias Pfisterer, M.D.*

*4. V.A. Medical Center, Long Beach and Cleveland Clinic Foundation:Robert Detrano, M.D., Ph.D.*

*5. Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.*

Formatted data from: http://archive.ics.uci.edu/ml/datasets/Heart+Disease
