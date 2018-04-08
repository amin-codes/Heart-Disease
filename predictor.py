#Made by: AKZOMBIE74 (Amin Zamani)
#Data: April 7, 2018
if __name__ == "__main__":

    #Importing some libraries
    import numpy as np
    import pandas as pd
    
    #Getting rid of pesky warnings
    def warn(*args, **kwargs):
        pass
    import warnings
    warnings.warn = warn
    np.warnings.filterwarnings('ignore')

    #START HERE
    column_names = [
            "age", #2
            "sex", #3
            "painloc", #4
            "painexer", #5
            "relrest", #6
            "resting-blood-pressure", #9
            "smoke", #12
            "famhist", #17
            "max-heart-rate-achieved", #31
            "heart-disease" #57
        ]

    #Importing the dataset
    location = 'longBeachVA.csv'
    dataset = pd.read_csv(location)
    X = dataset.iloc[:, [2, 3, 4, 5, 6, 9, 12, 17, 31]].values
    Y = dataset.iloc[:, 57].values
    
    #Replace all 'heart-disease' values greater than 0 because my goal is not to classify the disease type
    for x,i in enumerate(Y):
        if i>0:Y[x]=1

    #Taking care of missing data
    from sklearn.preprocessing import Imputer
    imputer = Imputer(missing_values=-9, strategy='most_frequent', axis=0)
    imputer.fit(X[:, [6,7]])
    X[:, [6,7]] = imputer.transform(X[:, [6,7]]) #Replace old data with new one.
    imputer = Imputer(missing_values=-9, strategy='mean', axis=0)
    imputer.fit(X[:, [5,8]])
    X[:, [5,8]] = imputer.transform(X[:, [5,8]])  # Replace old data with new one.

    #Splitting the dataset into the Training set and Test set
    from sklearn.model_selection._split import train_test_split
    from imblearn.combine import SMOTEENN
    smote_enn = SMOTEENN()
    X_resampled, y_resampled = smote_enn.fit_sample(X, Y)
    X_train, X_test, Y_Train, Y_Test = train_test_split(X_resampled, y_resampled, test_size=0.25)


    #Feature scaling
    from sklearn.preprocessing import StandardScaler
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)

    
    #Using Pipeline
    import sklearn.pipeline
    from sklearn.neural_network import MLPClassifier
    from sklearn.decomposition import KernelPCA
    from imblearn.pipeline import make_pipeline
    
    select = sklearn.feature_selection.SelectPercentile(sklearn.feature_selection.f_classif)
    clf = MLPClassifier(solver='lbfgs', learning_rate='constant', activation='tanh')
    kernel = KernelPCA()
    
    pipeline = make_pipeline(kernel, clf)
    pipeline.fit(X_train, Y_Train)
    
    #Testing
    #from sklearn import metrics
    #from sklearn.metrics import classification_report
    #y_pred = pipeline.predict(X_test)
    #report = metrics.classification_report(Y_Test, y_pred)
    #print report
    
    #User-input
    v = []
    for i in column_names[:-1]:
        v.append(input(i+": "))
    answer = np.array(v)
    answer = answer.reshape(1,-1)
    answer = sc_X.transform(answer)
    
    print "Predicts: " + str(pipeline.predict(answer))
    #If prediction == 0, then the person doesn't have heart-disease
    #else prediction == 1, then person has heart-disease
