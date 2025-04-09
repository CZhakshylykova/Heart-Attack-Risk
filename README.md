# Heart-Attack-Risk
Briefly about the dataset: 
data set downloaded from kaggle: https://www.kaggle.com/datasets/andrewmvd/heart-failure-clinical-data

I aimed to use this dataset to refresh my knowledge in statistics and brush up ML skills. 

# Overview of my data analysis
#1. already done - collected the data
#3. check 0 in the rows, identify outliers
#3. identify trends - plots histograms(try different kind of bins etc) or other visuals
    - handle outliers: I did with the zscore (abs. values > 3 are outliers)
#4. feature selection with RFE (recursive feature elimination)
#4. build a model (random forest classifier)
#5. evaluate the model performance
    - handle an overfitting