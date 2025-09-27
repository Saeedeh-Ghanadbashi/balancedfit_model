import pandas as pd
import numpy as np

df = pd.read_csv('./diabetes.csv')
df.head()

df.describe()

df.isnull().sum()

df.Outcome.value_counts()

X = df.drop('Outcome', axis='columns')
y = df.Outcome

from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
X_scaled = ss.fit_transform(X)

X[:3]

from sklearn.model_selection import train_test_split

train_x, test_x, train_y, test_y = train_test_split(X_scaled, y, random_state = 10, test_size = 0.3, stratify = y)

train_x.shape[0]
test_x.shape[0]

from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

scores = cross_val_score(DecisionTreeClassifier(), X, y, cv=5)
scores

from sklearn.ensemble import BaggingClassifier
scores = cross_val_score(BaggingClassifier(), X, y, cv=5)
scores

from sklearn.ensemble import RandomForestClassifier
scores = cross_val_score(RandomForestClassifier(), X, y, cv = 5)
scores

#The main reason why RandomForestClassifier often outperforms a plain BaggingClassifier is due to an extra layer of randomness introduced in Random Forests (random feature selection for each learner)