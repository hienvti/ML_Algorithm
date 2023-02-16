import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt


def PCA (X, num_components):
    # mean Centering the data  
    X_meaned = X - np.mean(X , axis = 0)
    # calculating the covariance matrix of the mean-centered data.
    cov_mat = np.cov(X_meaned , rowvar = False)

    # compute the eigenvalues and eigenvectors (tri rieng va vecto rieng)
    eigen_values, eigen_vectors = np.linalg.eigh(cov_mat)

    #sort the eigenvalues in descending order
    sorted_index = np.argsort(eigen_values)[::-1]
    sorted_eigenvalue = eigen_values[sorted_index]
    #similarly sort the eigenvectors 
    sorted_eigenvectors = eigen_vectors[:,sorted_index]

    # select the first n eigenvectors, n is desired dimension of our final reduced data.
    n_components = 2 #you can select any number of components.
    eigenvector_subset = sorted_eigenvectors[:,0:n_components]

    #Transform the data 
    X_reduced = np.dot(eigenvector_subset.transpose(),X_meaned.transpose()).transpose()

    return X_reduced

#Get the IRIS dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
data = pd.read_csv(url, names=['sepal length','sepal width','petal length','petal width','target'])
 
#prepare the data
x = data.iloc[:,0:4]
 
#prepare the target
target = data.iloc[:,4]
 
#Applying it to PCA function
mat_reduced = PCA(x , 2)
 
#Creating a Pandas DataFrame of reduced Dataset
principal_df = pd.DataFrame(mat_reduced , columns = ['PC1','PC2'])
 
#Concat it with target variable to create a complete Dataset
principal_df = pd.concat([principal_df , pd.DataFrame(target)] , axis = 1)

plt.figure(figsize = (6,6))
sb.scatterplot(data = principal_df , x = 'PC1',y = 'PC2' , hue = 'target' , s = 60 , palette= 'icefire')
plt.show()