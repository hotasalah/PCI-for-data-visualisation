import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

iris = datasets.load_iris()
x = pd.DataFrame( {'sepal length':iris.data[:,0], 'sepal wedth':iris.data[:,1], 'petal length':iris.data[:,2], 'petal wedth':iris.data[:,3]})
y = pd.DataFrame({'target':iris.target})

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1)

scaler = StandardScaler()
scaler.fit(x_train)

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

pca = PCA(n_components=2)

pca.fit(x_train)
PrincipalComponents = pca.transform(x_train)

pc = np.array(PrincipalComponents)
labels = np.array(y_train)

finalData = np.hstack((pc , labels))

finalDF = pd.DataFrame(finalData) 

print(finalDF)

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = [0, 1, 2]
colors = ['r', 'g', 'b']

for target, color in zip(targets,colors):
    indicesToKeep = finalDF[2] == target
    ax.scatter(finalDF.loc[indicesToKeep, 0]
               , finalDF.loc[indicesToKeep, 1]
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()


x_train = np.array( (finalDF.loc[:, 1].values).reshape(-1,1) )
y_train = np.array( (finalDF.loc[:,[2]].values).reshape(-1,1) )

model = LogisticRegression().fit(x_train, y_train)
r_sq = model.score(x_train, y_train)
print("coeficient of determination: ",r_sq)
y_pred = model.predict(x_test)
print(y_pred)


















