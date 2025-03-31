

#import the 
#.venv/bin/python -m pip install numpy # install the numpy into the virtual environment
#python -c "import numpy; print(numpy.__version__)" # verify the installent
#.venv/bin/python -m pip install pandas

#import the libraries
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt


#import the data
data  = pd.read_csv("heart_failure.csv")


#check the data structure
data.head()
data.describe
data.info()
data.dtypes
data.groupby("DEATH_EVENT").mean() #from there I see that people who had diabetes and deceased, were more advanced in age, having more severe aneamia and had less follow up
data.groupby("DEATH_EVENT").agg({"creatinine_phosphokinase":"mean", "diabetes":"median", "high_blood_pressure":"mean"}) #from there I see that people who had diabetes and deceased, were more advanced in age, having more severe aneamia and had less follow up
data[["age", "anaemia", "creatinine_phosphokinase"]]
data.columns


#plotting
plt.hist(data.anaemia, data.time)

