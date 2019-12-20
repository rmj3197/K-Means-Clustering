#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing required packages
import numpy as np
import scipy.io
import pandas as pd
from copy import deepcopy
from scipy.spatial import distance
import math
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# importing the dataset

# In[2]:


data=scipy.io.loadmat("C:/Users\Raktim\Desktop\ASU\SML\SML PROJECT 2\AllSamples.mat")['AllSamples']
#dataframes often come handy in many situations where working with numpy array is cumbersome or becomes difficult, so I am importing the data into a pandas dataframe too
df = pd.DataFrame({'Feature_1': data[:, 0], 'Feature_2': data[:, 1]})


# splitting the data column wise so that we can consider columns as x and y coordinates

# In[3]:


X, Y = data[:,0], data[:,1]


# defining the figure size for all of the plots to be used in this project

# In[4]:


fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 15
fig_size[1] = 10
plt.rcParams["figure.figsize"] = fig_size


# defining function for euclidean distance calculation

# In[5]:


def dist(x, y,ax=1):
    return np.linalg.norm(x - y,axis=ax)


# defining a list for strong the Within Cluster Sum of Squares (WCSS) and the index position of the datapoints selected as centroids using the farthest point approach

# In[6]:


index_list=[]
list_wcss=[]


# function for finding out the k different initial centroids
# the first centroid is randomly chosen and the next centroids are chosen such that the average distance of this chosen one to all previous (i-1) centers is maximal.

# In[7]:


def initial_centroids(k):
    centroids = []
    d = np.zeros([len(data), k-1])
    
    #first centroid is initialized randomly
    random = np.random.choice(data.shape[0], 1, replace = False)
    centroid_1 = data[random]
    index_list.append(random[0])
    centroids.append(data[random][0])
  
    for i in range(k-1):
        d[:,i] = dist(centroids[i],data, ax = 1)
        dist_mean = np.mean(d[:,:i+1], axis=1)
        index = np.argmax(dist_mean)
        
        for j in range(0,len(index_list)):
            if index in index_list:
                dist_mean[index]=-1
                index=np.argmax(dist_mean)
        index_list.append(index)
        centroids.append(data[index])
    return centroids
# centroids=np.reshape(initial_centroids(10),(10,2))


# defining the K-Means Function which has parameters the number of clusters and the centroids that have been determined by the maximal distance approach

# In[8]:


def kmeans (k):
    
    centroid=np.reshape(initial_centroids(k),(k,2))
    print('The centroids generated by the maximal distance approach are\n',centroid)
    
    #scatter plot of data and the randomly chosen centroids
    plt.scatter(X, Y, c='black')
    plt.scatter(centroid[:,0], centroid[:,1], marker='D', s=100, c='red')
    
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
    ax.scatter(centroid[:, 0], centroid[:, 1], marker='X', s=100, c='black')
    
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

# In[9]:


kmeans(2)


# In[10]:


kmeans(3)


# In[11]:


kmeans(4)


# In[12]:


kmeans(5)


# In[13]:


kmeans(6)


# In[14]:


kmeans(7)


# In[15]:


kmeans(8)


# In[16]:


kmeans(9)


# In[17]:


kmeans(10)


# Plotting Objective Function vs Number of Cluster 

# In[18]:


#list storing all the values of k
k_val=[2,3,4,5,6,7,8,9,10]


# In[19]:


#plotting the objective function
plt.plot(k_val,list_wcss)
plt.scatter(k_val,list_wcss)
plt.grid()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:



