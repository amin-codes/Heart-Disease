# :heart: Helper
*Note: Only longBeachVA.csv is used for training this model*

This project is an attempt at making a machine-learning model that can predict whether a person has heart disease or not without having to visit a doctor through the following inputs:
* age (integer)
* sex (1 = male; 0 = female)
* chest pain location (1 = substernal; 0 = otherwise) 
* pain is provoked by exertion? (1 = provoked by exertion; 0 = otherwise)
* relieved after rest? (1 = relieved after rest; 0 = otherwise) 
* resting blood pressure (in mm Hg on admission to the hospital) (systolic) (If your blood pressure is 120/80, then enter 120)
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
# About the Model
The model's core is comprised of a [pipeline](http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) that includes a [Kernel PCA](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.KernelPCA.html) followed by a [MLP Classifier](http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html).

The MLP Classifier uses a LBFGS solver and a tanh activation. 

...More to be added later
# Requirements to Run Program
* Install Python (version 2.7.14)
* Install packages: numpy (version >= 1.14.2), pandas (version >= 0.21), scikit-learn (version >= 0.19.1), imbalanced-learn (version >= 0.3.3)

# References
*1. Hungarian Institute of Cardiology. Budapest: Andras Janosi, M.D.*

*2. University Hospital, Zurich, Switzerland: William Steinbrunn, M.D.*

*3. University Hospital, Basel, Switzerland: Matthias Pfisterer, M.D.*

*4. V.A. Medical Center, Long Beach and Cleveland Clinic Foundation:Robert Detrano, M.D., Ph.D.*

Formatted data from: http://archive.ics.uci.edu/ml/datasets/Heart+Disease
