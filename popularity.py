
import numpy as np
import pandas as pd
import math
import json
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
import joblib as jb
import scipy.sparse
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
import warnings; warnings.simplefilter('ignore')
import pickle

# Import the dataset and give the column names
columns=['userId', 'productId', 'ratings']
products = pd.read_csv('moblie.csv')
products.columns = columns

products.head()



#Check the number of rows and columns
rows,columns=products.shape
print('Number of rows: ',rows)
print('Number of columns: ',columns)

#Check the datatypes
products.dtypes

#Taking subset of the dataset
products1=products.iloc[:200,0:]

products1.info()

#Summary statistics of rating variable
products['ratings'].describe().transpose()

#Find the minimum and maximum ratings
print('Minimum rating is: %d' %(products1.ratings.min()))
print('Maximum rating is: %d' %(products1.ratings.max()))


#Check for missing values
print('Number of missing values across columns: \n',products.isnull().sum())


# Check the distribution of the rating
with sns.axes_style('white'):
    g = sns.factorplot("ratings", data=products1, aspect=2.0,kind='count')
    g.set_ylabels("Total number of ratings")


# Number of unique user id  in the data
print('Number of unique users in Raw data = ', products1['userId'].nunique())
# Number of unique product id  in the data
print('Number of unique product in Raw data = ', products1['productId'].nunique())


#Check the top 10 users based on ratings
most_rated=products1.groupby('userId').size().sort_values(ascending=False)[:10]
print('Top 10 users based on ratings: \n',most_rated)


#Check the top 10 users based on ratings
most_rated_p=products1.groupby('productId').size().sort_values(ascending=False)[:10]
print('Top 10 users based on ratings: \n',most_rated_p)


counts=products1.userId.value_counts()
products1_final=products1[products1.userId.isin(counts[counts>=2].index)]
print('Number of users who have rated 25 or more items =', len(products1_final))
print('Number of unique users in the final data = ', products1_final['userId'].nunique())
print('Number of unique products in the final data = ', products1_final['userId'].nunique())


#constructing the pivot table
final_ratings_matrix = products1_final.pivot(index = 'userId', columns ='productId', values = 'ratings').fillna(0)
final_ratings_matrix.head()

print('Shape of final_ratings_matrix: ', final_ratings_matrix.shape)


#Calucating the density of the rating marix
given_num_of_ratings = np.count_nonzero(final_ratings_matrix)
print('given_num_of_ratings = ', given_num_of_ratings)
possible_num_of_ratings = final_ratings_matrix.shape[0] * final_ratings_matrix.shape[1]
print('possible_num_of_ratings = ', possible_num_of_ratings)
density = (given_num_of_ratings/possible_num_of_ratings)
density *= 100
print ('density: {:4.2f}%'.format(density))

#Split the data randomnly into train and test datasets into 70:30 ratio
train_data, test_data = train_test_split(products1_final, test_size = 0.3, random_state=0)
train_data.head()

print('Shape of training data: ',train_data.shape)
print('Shape of testing data: ',test_data.shape)

#Count of user_id for each unique product as recommendation score 
train_data_grouped = train_data.groupby('productId').agg({'userId': 'count'}).reset_index()
train_data_grouped.rename(columns = {'userId': 'score'},inplace=True)
train_data_grouped.head(40)


#Sort the products on recommendation score 
train_data_sort = train_data_grouped.sort_values(['score', 'productId'], ascending = [0,1]) 
      
#Generate a recommendation rank based upon score 
train_data_sort['rank'] = train_data_sort['score'].rank(ascending=0, method='first') 
          
#Get the top 5 recommendations 
popularity_recommendations = train_data_sort.head(10) 
popularity_recommendations

# Use popularity based recommender model to make predictions

def recommend(user_id):     
    user_recommendations = popularity_recommendations 
          
    #Add user_id column for which the recommendations are being generated 
    user_recommendations['userId'] = user_id 
      
    #Bring user_id column to the front 
    cols = user_recommendations.columns.tolist() 
    cols = cols[-1:] + cols[:-1] 
    user_recommendations = user_recommendations[cols] 
    predictions = pd.DataFrame(popularity_recommendations,columns=['productId'])
    pred = predictions['productId'].tolist()
    
          
    return pred





model = pickle.dump(popularity_recommendations,open('popular.pkl', 'wb'))