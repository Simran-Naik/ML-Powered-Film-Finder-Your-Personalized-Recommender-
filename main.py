#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


credits_file_path = r"C:\Users\simra\OneDrive\Desktop\MACHINE LEARNING for tmdb_5000_credits\tmdb_5000_credits.csv"
credits = pd.read_csv(credits_file_path)
movies_file_path = r"C:\Users\simra\OneDrive\Desktop\MACHINE LEARNING for tmdb_5000_movies\tmdb_5000_movies.csv"
movies = pd.read_csv(movies_file_path)


# In[3]:


movies.head(1)


# In[4]:


credits.head(1)


# In[5]:


credits.head(1)['crew'].values


# In[6]:


movies=movies.merge(credits,on="title")


# In[7]:


movies.head()


# In[8]:


#columns not required ,cleaning we need to create tags for each movie so that 
#recommendation should work,columns not relevant are budget column,runtime,
#homepage,original language,original title(title is already kept ),popularity
#production companies,status,production_countries,crew,release_date,revenue,
#spoken_languages,tagline,vote_average,vote_count


# In[9]:


movies.info()
#important ones
#genres,title ,cast,crew,id,keywords


# In[10]:


movies=movies[["movie_id","title","overview","genres","keywords","cast","crew"]]


# In[11]:


movies.info()


# In[12]:


movies.head()


# In[13]:


#to check missing/duplicate data
movies.isnull().sum()


# In[14]:


movies.dropna(inplace=True)


# In[15]:


movies.isnull().sum()#to check null/missing values


# In[16]:


movies.duplicated().sum()#to check duplicates


# In[17]:


movies.iloc[0].genres #convert list of dictionaries of string to  string 
#in list 
import ast


# In[18]:


#make a function to convert it string of  list into list use module ast,
#makes a list and literal_eval is a function in the module of ast 
def convert(obj):
    L=[]
    for i in ast.literal_eval(obj):
        L.append(i["name"])
    return L


# In[19]:


movies["genres"]= movies["genres"].apply(convert)


# In[20]:


movies.head()#here we have the dictionary list string in genres to string 
#in list do the same for keywords,cast,crew


# In[21]:


movies["keywords"]=movies["keywords"].apply(convert)


# In[22]:


movies.head()


# In[23]:


#in cast we need only 3 top actors we dont need all the actors,we need 1st
#3 dictionary which castid,character,creditid
def convert3(obj):
    L=[]
    counter = 0
    for i in ast.literal_eval(obj):
        if counter != 3:
            L.append(i["name"])
            counter+=1
        else:
            break
    return L
        


# In[24]:


movies["cast"]=movies['cast'].apply(convert3)


# In[25]:


movies.head()


# In[26]:


movies["crew"][0]


# In[27]:


#for crew ,i need dictionary that job=director,i need to extarct that name
#job value is director,make a function
# Assuming movies is your DataFrame
#def fetch_director(obj):
    #L=[]
    #for i in json.loads(obj):
        #if i["job"] == "Director":
           # L.append(i["name"])
# break
#import json
#def fetch_director(obj):
   # L=[]
    #for i in json.loads(obj):
     #   crew=[L["name"] for L in i if L.get("job") == "Director"]
    #return crew


# In[28]:


import json
#Assuming movies is your DataFrame
def fetch_director(obj):
    crew_list = json.loads(obj)  # Parse the JSON-like string to a list of dictionaries
    crew = [crew_member["name"] for crew_member in crew_list if crew_member.get("job") == "Director"]
    return crew
   


# In[29]:


movies["crew"] = movies["crew"].apply(fetch_director)


# In[30]:


movies.head()


# In[31]:


#convert overview colum string to list,to concatenate with other list
movies["overview"][0]


# In[32]:


movies["overview"]=movies["overview"].apply(lambda x:x.split())
#every row has a list now


# In[33]:


movies.head()


# In[34]:


# we need to concatenate genres,keywords,cast and crew to get a big list 
#we need to apply transformation,we need to remove space in between strings
#sam worthington to samworthington, beacuse we have 2 words sam worthington of
#same person and then we have sam mendes 2 words of the different person
#here 2 persons have the same first name so if we connect first and last name\
#then we could get the correct name ,so i want to see san mendes moview
#so the model will get confused which movies should it show
# we make one word as sammendes or samworthington we wehn we search for sam
#mendes we get the correct name
movies['genres']=movies['genres'].apply(lambda x :[i.replace(" ","") for i in x])
movies['keywords']=movies['keywords'].apply(lambda x :[i.replace(" ","") for i in x])
movies['cast']=movies['cast'].apply(lambda x :[i.replace(" ","") for i in x])
movies['crew']=movies['crew'].apply(lambda x :[i.replace(" ","") for i in x])


# In[35]:


movies.head()


# In[36]:


#we need to conctenate generes,overview,cast,crew to make a new colum tags
movies['tags']= movies['overview'] + movies['keywords']+ movies['cast']+movies['crew']


# In[37]:


movies.head()


# In[38]:


# to remove the overview,generes,cast and crew column
new_df=movies[["movie_id","title","tags"]]


# In[39]:


new_df.head()


# In[40]:


#convert list to string in tags
new_df['tags']=new_df['tags'].apply(lambda x:" ".join(x))


# In[41]:


new_df.head()


# In[42]:


new_df['tags'][0]#u can se the overview,geners,keywords,cast and crew


# In[43]:


#convert it to lower case all the letters
new_df['tags']=new_df['tags'].apply(lambda x:x.lower())


# In[44]:


new_df.head()


# In[45]:


#MACHINE LEARNING
#I NEED SIMILARITY BETWEEN 2 TAGS,SIMILARITY BETWEEN 2 TEXTS,BEACUSE OUR 
#MAIN OBJECTIVE IS TO CREATE A RECOMMENDATION SYSTEM FOR A PERSON WHEN 
# HE TYPES A MOVIE HE SHOULD BE RECOMMENDED A SIMILAR MOVIE
#WE COULD SEE WHAT WORDS ARE SIMILAR IN BOTH THE TAGS BUT THAT WOULD NOT HELP
#ITS NOT RELEVANT,SO WE CHANGE THE TEXT INTO VECTORS AND THIS IS VECTORIZATION
#WE HAVE MOVIEID AND NAME AND TAGS ,WE HAVE ALMOST 5000 MOVIES SO CONVERT ALL TAGS TO VECTORS
#EVERY TEXT BE CONVERTED TO VECTOR EVERY MOVIE BECOMES A VECTOR ,SO IF THE PERSON
#CHOOSES A MOVIE HE CHOSES ONE VECTORE THEN WE RECOMMEND THE VECORS NEAR IT TO THE PERSON
# WE USE BAG OF WORDS VECTORIZAION
#ALL THE TAGS ARE COMBINED HERE ,THEN WE CHOOSE THE WORDS WHICH ARE USED MANY
#TIMES OR MOST COMMON WORDS USED IN THIS COMBINATION,EXTRACT THEM  THEN IN 
#THE FIST TAG I WILL CHECK THE MOST COMMON WORD HOW MANY TIMES HAS IT 
#REPEATED IN 1ST TAG SO MAKE A TABLE,USE THE SAME WORD AND CHECK IN THE 2ND TAG
#MAKE A TABLE,TAKE THE SAME WORD AND CHECK IN 3RD TAG MAKE A TABLE
#THIS TABLE HAS BECOME A VECTOR EACH MOVIE HAS A VECTOR TABLE OF MOST COMMON WORDS USES
#NO OF MOVIES AND THE WORDS AND NO OF TIMES IT IS REPEATED IS PLOTTED IN THE
#XY CHART NOW WHEN U TYPE A MOVIE THAT MOVIE SEE WHAT OTHER MOVIE HAS THE 
#SAME REPEATED WORDS AND RECOMMENDED TO  THEM ,STOPWORDS ARE NOT 
#CONSIDERED ARE,OR,IS,AND,A,TO,IT,THE

new_df['tags'][0]


# In[46]:


#2nd tag
new_df['tags'][1]


# In[47]:


#IN SCIKET LEARN WE HAVE A FUNCTION COUNTVECTORIZER FOR VECTORIZATION
#max_feature tell how many words u will take,stopwords none in english
from sklearn.feature_extraction.text  import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words="english")


# In[48]:


#convert tags to numpy array to make it into vector
vectors=cv.fit_transform(new_df['tags']).toarray()


# In[49]:


vectors


# In[50]:


vectors[0]#this is the first movie avatar


# In[51]:


#to see what is the most 5000 frequent words
cv.get_feature_names()
#list of 5000 items words
#corpus=combination os tags into a massive string
#but here we can see the words are repeating like action,activity,ue steaming
#it removes words which are extension of orginal word loving,loved


# In[52]:


len(cv.get_feature_names())


# In[53]:


get_ipython().system('pip install nltk')
#this is to use steaming


# In[54]:


from nltk.stem.porter import PorterStemmer
ps =PorterStemmer()


# In[55]:


def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)      
#this is a helper function loved becomes love


# In[56]:


ps.stem('acting')


# In[57]:


stem("captain barbossa, long believed to be dead, has come back to life and is headed to the edge of the earth with will turner and elizabeth swann. but nothing is quite as it seems. ocean drugabuse exoticisland eastindiatradingcompany loveofone'slife traitor shipwreck strongwoman ship alliance calypso afterlife fighter pirate swashbuckler aftercreditsstinger johnnydepp orlandobloom keiraknightley goreverbinski")


# In[58]:


new_df['tags']=new_df['tags'].apply(stem)


# In[59]:


cv.get_feature_names()
#u can see the words are not repeating


# In[60]:


#4806 movies in total ,4806 vectors and each vectors have 5000 number
#every movie we need to calculate the distnce ,less distance between movie
#more comman,calcualte cosine distance,it calculates the angle between them
#if angle is 5 between 2 movies very comman movie and if its 180 degrss
#then the distance is large and the movies are different
#euclidean distance is tip of line to tip of other line distance
#euclidean distance is bad measure for high dimension data 2d ,3d
#calculate the distance get to know the recommendation between movies
#distance is inversly proportional to similarity
from sklearn.metrics.pairwise import cosine_similarity


# In[61]:


cosine_similarity(vectors)#distance between each movies 54806*4806


# In[62]:


similarity=cosine_similarity(vectors)


# In[63]:


similarity[0]#1stmovie similarity/distance to other movies,
#the similarity is 1 in the ist avlue as the similarity with the ist movie to 
#ist movie is obviously 1
#avatar distance with other movies


# In[64]:


#make a function which would recommend movie looking at the similarity
#movie given ,i need to finds its index then using index find similairy 
#and take that array in which similarity of that movie is given 
#with other movies,sort the similarites or distance in that array in asc
#order,lest similar last and more similar on top,top similar are top 5 movies
#use index to get inside the table 

def recommend(movie):
    movie_index = new_df[new_df['title']== movie].index[0]#to find the index 
#of the movie asked to recommend
#now find similaity or distance,#explanation in nelow 4 codes
    distances = similarity[movie_index]
#sort the distance for the movie asked in asc
    movies_list=sorted(list(enumerate(similarity[0])),reverse=True,key=lambda x:x[1])[1:6]
       
    for i in movies_list:
        print(new_df.iloc[i[0]].title)
       
    


# In[65]:


new_df[new_df['title']=='Batman Begins'].index[0]


# In[66]:


sorted(similarity[0],reverse = True)
#shows the movies which are most similar ,but in this index positions are getting lost
#so the function recommend would fail as here when we are sorting the index is
#changed,we need to keep the index positon intact using enumerate function
#


# In[67]:


list(enumerate(similarity[0]))
#this gives index is it list of tuples and idstances between the movies
#oth movie distance with 0 th movie is 1 0th movie with 1st movie is 0


# In[68]:


#sorting the movie when the index function is fixed 
#sort the list keeping the index position intact and show from the last 
#reverse true then use lambda to tell the funciton to use sorting only on 2 nd
#position not on ist colum which is index and i want between 1 to 5
#top 5 movies whose distance are closer to 1st movie
sorted(list(enumerate(similarity[0])),reverse=True,key=lambda x:x[1])[1:6]


# In[69]:


recommend('Avatar')
#here it is recomending the index only i want it to recommend movie only


# In[70]:


new_df.iloc[1216].title


# In[71]:


recommend("Batman Begins")


# In[72]:


#use pycharm to make website


# In[73]:


import pickle


# In[74]:


pickle.dump(new_df,open('movies.pkl','wb'))


# In[75]:


new_df['title'].values


# In[76]:


pickle.dump(new_df.to_dict(),open('movie_dict.pkl','wb'))#rather than sending the dataframe we send the dictionary


# In[77]:


pickle.dump(similarity,open('similarity.pkl','wb'))


# In[78]:


recommend("Batman Begins")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




