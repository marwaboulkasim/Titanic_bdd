import pandas as pd
import plotly as px
import seaborn as sns
import matplotlib.pyplot as plt

# Chargement des données
#df = pd.read_csv('train.csv')
#df.head()
#df.info()
#df.describe()
#df.isnull().sum()

#train = pd.read_csv("train.csv")
#test = pd.read_csv("test.csv")
#def dataprep(data):
 #   sexe = pd.get_dummies(data['Sex'], prefix='sex')

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
train.head()
train.info()
train.describe()
train.isnull().sum()

  
print(train['Survived'].value_counts(normalize=True))
print(train.groupby('Sex')['Survived'].mean())
print(train .groupby('Pclass')['Survived'].mean())
print(train['Age'].isnull().sum())
train.describe()




