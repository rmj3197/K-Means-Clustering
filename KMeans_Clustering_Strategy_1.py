#!/usr/bin/env python
# coding: utf-8

# In[21]:


#importing required packages
import numpy as np
import scipy.io
import pandas as pd
from copy import deepcopy
import math
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# importing the dataset for this project

# In[22]:


data=scipy.io.loadmat("C:/Users\Raktim\Desktop\ASU\SML\SML PROJECT 2\AllSamples.mat")['AllSamples']

#dataframes often come handy in many situations where working with numpy array is cumbersome or becomes difficult, so I am importing the data into a pandas dataframe too
dataframe = pd.DataFrame({'Feature_1': data[:, 0], 'Feature_2': data[:, 1]})


# splitting the data column wise so that we can consider columns as x and y coordinates

# In[23]:


X, Y = data[:,0], data[:,1]


# defining the figure size for all of the plots to be used in this project

# In[24]:


fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 15
fig_size[1] = 10
plt.rcParams["figure.figsize"] = fig_size


# plotting the data

# In[25]:


plt.scatter(X,Y,s=100,c='black')
plt.grid()


# defining function for euclidean distance calculation

# In[26]:


def dist(x, y,ax=1):
    return np.linalg.norm(x - y,axis=ax)


# defining a list for strong the Within Cluster Sum of Squares (WCSS)

# In[27]:


list_wcss=[]


# defining the K-Means Function

# In[28]:


def kmeans (k):
    
    #randomly choosing a sample from the pandas dataframe
    centroid=(dataframe.sample(n=k)).to_numpy()
    print('Initial Randomly chosen Centroids are\n',centroid)
    
    #scatter plot of data and the randomly chosen centroids
    plt.scatter(X, Y, c='black')
    plt.scatter(centroid[:,0], centroid[:,1], marker='D', s=300, c='red')
    
     # for storing the old values of the centroids when they are updated 
    centroid_old = np.zeros(centroid.shape)
    
    # Labelling the clusters with suitable label - 0/1/2/3/4/5/6/7/8/9
    clusters = np.zeros(len(data))
    
    # Distance between new centroids and old centroids - this is imporatnt because loop will run till this distance becomes zero
    diff = dist(centroid, centroid_old, None)
    
    # Loop will run till the diff becomes zero
    while diff != 0:
        # Assigning each value to its closest cluster
        for i in range(len(data)):
            distances = dist(data[i], centroid)
            cluster = np.argmin(distances)
            clusters[i] = cluster
        # Storing the old centroid values
        centroid_old = deepcopy(centroid)
        
        # Finding the new centroids by taking the average value
        for i in range(k): 
            points = [data[j] for j in range(len(data)) if clusters[j] == i]
            centroid[i] = np.mean(points, axis=0)
        diff = dist(centroid, centroid_old, None)
    fig, ax = plt.subplots()
    for i in range(k):
            points = np.array([data[j] for j in range(len(data)) if clusters[j] == i])
            ax.scatter(points[:, 0], points[:, 1])
    ax.scatter(centroid[:, 0], centroid[:, 1], marker='X', s=500, c='black')
    
    #printing the final centroids after running the iterations
    print('The final centroids are\n',centroid)
    
    #adding the cluster label to the original data
    data_mod=np.insert(data,2,clusters,axis=1)
    
    #arranging the data in order of increasing value of data cluster id
    data_cluster_label_sorted=data_mod[data_mod[:,2].argsort(kind='mergesort')]
    
    #dataframe with the cluster label
    dataframe_cluster_label=pd.DataFrame({'Feature_1': data_cluster_label_sorted[:, 0], 'Feature_2': data_cluster_label_sorted[:, 1],'Cluster_Label':data_cluster_label_sorted[:,2]})
    
    #splliting the dataset into various subsets according to the cluster label
    subset={}
    for j in range (0,dataframe_cluster_label['Cluster_Label'].nunique()):
        subset[j]=(dataframe_cluster_label[dataframe_cluster_label['Cluster_Label']==j].drop('Cluster_Label',1)).to_numpy()
    
    #calculating the value of objective function/wcss
    wcss=0
    for j in range (0,dataframe_cluster_label['Cluster_Label'].nunique()):
        for k in range (0,len(subset[j])):
            wcss+=math.pow(dist(centroid[j],subset[j][k], None),2)
    list_wcss.append(wcss)
    print('WCSS is',wcss)


# Clustering for K=2 to K=10

# In[29]:


kmeans(2)


# In[30]:


kmeans(3)


# In[31]:


kmeans(4)


# In[32]:


kmeans(5)


# In[33]:


kmeans(6)


# In[34]:


kmeans(7)


# In[35]:


kmeans(8)


# In[36]:


kmeans(9)


# In[37]:


kmeans(10)


# OBJECTIVE FUNCTION VS NUMBER OF CLUSTERS (ELBOW) PLOT

# In[38]:


#list storing the number of clusters
k_val=[2,3,4,5,6,7,8,9,10]


# In[39]:


plt.plot(k_val,list_wcss)
plt.scatter(k_val,list_wcss)
plt.grid()


# In[40]:


list_wcss


# In[ ]:





# In[ ]:





# In[ ]:




