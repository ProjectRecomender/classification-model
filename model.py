import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing

#read dataset and prepare it
df = pd.read_csv('careers.csv')
df.drop("Timestamp", axis=1, inplace = True)


X = df[["You have an eye for design", "You have attention to detail", "You like to identify new ways to improve a company or project", "You like to analyze data", "You like to see or discover patterns", "You like to manage projects", "You like to lead"]]
y = df["Which one of the listed careers interest you the most?"].values

#split the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 4)

from sklearn.linear_model import LogisticRegression
LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train,y_train)

yhat = LR.predict(X_test)
yhat_prob = LR.predict_proba(X_test)