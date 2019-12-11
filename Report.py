#!/usr/bin/env python
# coding: utf-8

# # Reading Data

# In[2]:


#importing the needed libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gc
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


train_data = pd.read_csv('train.csv')
resources_data = pd.read_csv("resources.csv")
#test_data = pd.read_csv('test.csv')


# In[129]:


#test_data.head()


# # Data Discovery

# #### Taking a peak at the training data

# In[3]:


train_data.head()


# We can see that most of our features are categotical or text so we will probably need lots of feature engineering. Also we can see that there's a possible lots of NaNs in essay 3 & 4, for further inspection lets use describe.

# In[4]:


train_data.describe()


# In[5]:


train_data.describe(include = ['O'])


# #### Taking a peak at the resources data

# In[6]:


resources_data.head(10)


# In[7]:


print(train_data.where(train_data['id']=='p096795').dropna())


# In[8]:


resources_data.describe()


# In[9]:


resources_data.describe(include=['O'])


# #### Checking missing values

# In[10]:


(train_data.isnull().sum()/len(train_data))*100


# - In training data, we can see that project_essay_4 and project_essay_3 having 96 % null values. so during prediction, better remove these 2 columns.

# In[11]:


(resources_data.isnull().sum()/len(train_data))*100


# - In resource data, only description column having few null values. So we can ignore these values.

# #### Checking Data balance

# In[12]:


approvedPercentage=(train_data['project_is_approved'].sum() / len(train_data['project_is_approved']))*100
# Pie chart, where the slices will be ordered and plotted counter-clockwise:
labels = 'Accepted', 'Rejected'
sizes = [approvedPercentage, 100 - approvedPercentage]
plt.figure(figsize=(16,6))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True, startangle=181)
plt.show()
del approvedPercentage


# - Training data is highly imbalanced that is approx. 85 % projetcs were approved and 15 % project were not approved. Majority imbalanced class is positive.

# #### Histograms & Distributions of some of the features

# In[13]:


plt.figure(figsize=(16,6))
histData=train_data['school_state'].value_counts()
values = (histData.values/train_data['school_state'].value_counts().sum())*100
plt.bar(histData.index,values)
plt.title('Distribution of School states in %')
plt.xlabel('States')
plt.ylabel('Count Percentage')
plt.show()
del histData
del values


# - Out of 50 states, California(CA) having higher number of projects proposal submitted approx. 14 % followed by Texas(TX)(7 %) and Tennessee(NY)(7 %).

# In[14]:


plt.figure(figsize=(16,6))
histData=train_data['project_grade_category'].value_counts()
values = (histData.values/train_data['project_grade_category'].value_counts().sum())*100
plt.bar(histData.index,values)
plt.title('Distribution of Grade Categories in %')
plt.xlabel('Grade Categories')
plt.ylabel('Count Percentage')
plt.show()
del histData
del values


# - Out of 4 school grade levels, Project proposals submission in school grade levels is higher for Grades Prek-2 which is approximately 41 % followed by Grades 3-5 which has approx. 34 %.

# In[15]:


plt.figure(figsize=(18,7))
histData=train_data['project_subject_categories'].value_counts().iloc[0:11]
values = (histData.values/train_data['project_subject_categories'].value_counts().sum())*100
plt.bar(histData.index,values)
plt.title('Distribution of Subject Categories in %')
plt.xlabel('Grade Categories')
plt.ylabel('Count Percentage')
plt.show()
del histData
del values


# - Out of 51 Project categories, Project proposals submission for project categories is higher for Literacy & Language which is approx. 21.5 % followed by Math & Science which has approx. 15.7 %.

# In[16]:


plt.figure(figsize=(16,7))
plt.hist(train_data['teacher_number_of_previously_posted_projects'],color="skyblue")
plt.title('Distribution of number of teachers\' previously posted projects')
plt.xlabel('Number of teachers\' previously posted projects')
plt.ylabel('Count')
plt.show()


# In[17]:


plt.figure(figsize=(18,7))
histData=train_data['project_subject_subcategories'].value_counts().iloc[0:11]
values = (histData.values/train_data['project_subject_subcategories'].value_counts().sum())*100
plt.bar(histData.index,values)
plt.title('Distribution of Subject Sub-categories in %')
plt.xlabel('Grade Categories')
plt.ylabel('Count Percentage')
plt.show()
del histData
del values


# - Out of 182,020 Project subcategories, Project proposals submission for project sub-categoriesis is higher for Literacy which is approx. 9 % followed by Literacy & Mathematics which has approx. 8 % .

# In[18]:


plt.figure(figsize=(18,7))
histData=train_data['project_title'].value_counts().iloc[0:11]
values = (histData.values/train_data['project_title'].value_counts().sum())*100
plt.bar(histData.index,values)
plt.title('Distribution of Project titles in %')
plt.xlabel('Project title')
plt.ylabel('Count Percentage')
plt.show()
del histData
del values


# - Out of 182,080 project titles, Project proposals submission for project titles is higher for Flexible seating which is approx. 21 % followed by Whiggle while you work which has approx. 8 %. But we can see that there are slightly different variations of Wiggle While You Work which will add up to about 30% and of Flexible Seating which will add up to about 24%

# In[19]:


plt.figure(figsize=(16,7))
plt.hist(resources_data['price'])
plt.title('Distribution of resources prices')
plt.xlabel('Resources Price')
plt.ylabel('Count')
plt.show()


# - There are some items whose price is 0 which looks a little weird but lets hold that thought.
# - As we can see most of the price requested for resources is between 0 to 2k dollar.

# In[20]:


plt.figure(figsize=(16,7))
plt.hist(resources_data['quantity'])
plt.title('Distribution of resources quantity')
plt.xlabel('Resources quantity')
plt.ylabel('Count')
plt.show()


# - There are some items whose quantity is 0 which looks a little weird but lets hold that thought.

# In[21]:


plt.figure(figsize=(16,7))
plt.hist(train_data['teacher_prefix'].dropna())
plt.title('Distribution of resources quantity')
plt.xlabel('Resources quantity')
plt.ylabel('Count')
plt.show()


# - Higher number of project proposals are submitted by married women which is approx. 53 % followed by unmarried women which has approx. 37 %.
# - Project proposal submitted by Teacher which is approx. 2 % is vey low as compared to Mrs., Ms., Mr.
# - There are very few projects submitted by Drs.

# ### Wordclouds

# In[59]:


from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator


# #### Resources summary

# In[23]:


text=train_data['project_resource_summary'].values
wordcloud = WordCloud().generate(str(text))
plt.figure(figsize=(12,6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
del text
del wordcloud


# #### Resources description

# In[24]:


text=resources_data['description'].values
wordcloud = WordCloud().generate(str(text))
plt.figure(figsize=(12,6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
del text
del wordcloud


# #### Here we compare the word clouds of accepted & rejected proposals for the first essay

# In[25]:


text=train_data.where(train_data['project_is_approved']==1)['project_essay_1'].values
wordcloud = WordCloud().generate(str(text))
plt.figure(figsize=(12,6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
del text
del wordcloud


# In[26]:


text=train_data.where(train_data['project_is_approved']==0)['project_essay_1'].values
wordcloud = WordCloud().generate(str(text))
plt.figure(figsize=(12,6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
del text
del wordcloud


# - We can see that there are common words that should be execluded like student, students & school.
# - We can see that lots of rejected projects leave the essay empty (nan)

# #### Here we compare the word clouds of accepted & rejected proposals for the second essay

# In[27]:


text=train_data.where(train_data['project_is_approved']==1)['project_essay_2'].values
wordcloud = WordCloud().generate(str(text))
plt.figure(figsize=(12,6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
del text
del wordcloud


# In[28]:


text=train_data.where(train_data['project_is_approved']==0)['project_essay_2'].values
wordcloud = WordCloud().generate(str(text))
plt.figure(figsize=(12,6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
del text
del wordcloud


# - Student & will exist in both accepted & rejected nearly equally

# #### Acceptance Vs Rejection across Different features

# In[29]:


plt.figure(figsize=(18,7))
histData=train_data['school_state'].value_counts().iloc[0:22]
acceptedValues = train_data.where(train_data['project_is_approved']==1)['school_state'].value_counts().iloc[0:22].values
rejectedValues = train_data.where(train_data['project_is_approved']==0)['school_state'].value_counts().iloc[0:22].values
plt.bar(histData.index,acceptedValues)
plt.bar(histData.index,rejectedValues,bottom=acceptedValues)
plt.title('Accepted & Rejected projects Vs states')
plt.xlabel('State')
plt.ylabel('Count')
plt.show()
del histData
del acceptedValues
del rejectedValues


# In[30]:


plt.figure(figsize=(18,7))
histData=train_data['teacher_prefix'].value_counts()
acceptedValues = train_data.where(train_data['project_is_approved']==1)['teacher_prefix'].value_counts().values
rejectedValues = train_data.where(train_data['project_is_approved']==0)['teacher_prefix'].value_counts().values
plt.bar(histData.index,acceptedValues)
plt.bar(histData.index,rejectedValues,bottom=acceptedValues)
plt.title('Accepted & Rejected projects Vs Prefixes')
plt.xlabel('Prefix')
plt.ylabel('Count')
plt.show()
del histData
del acceptedValues
del rejectedValues


# In[31]:


plt.figure(figsize=(18,7))
histData=train_data['project_grade_category'].value_counts()
acceptedValues = train_data.where(train_data['project_is_approved']==1)['project_grade_category'].value_counts().values
rejectedValues = train_data.where(train_data['project_is_approved']==0)['project_grade_category'].value_counts().values
plt.bar(histData.index,acceptedValues)
plt.bar(histData.index,rejectedValues,bottom=acceptedValues)
plt.title('Accepted & Rejected projects Vs Grades')
plt.xlabel('Grade')
plt.ylabel('Count')
plt.show()
del histData
del acceptedValues
del rejectedValues


# In[32]:


plt.figure(figsize=(18,7))
histData=train_data['project_subject_categories'].value_counts().iloc[0:11]
acceptedValues = train_data.where(train_data['project_is_approved']==1)['project_subject_categories'].value_counts().iloc[0:11].values
rejectedValues = train_data.where(train_data['project_is_approved']==0)['project_subject_categories'].value_counts().iloc[0:11].values
plt.bar(histData.index,acceptedValues)
plt.bar(histData.index,rejectedValues,bottom=acceptedValues)
plt.title('Accepted & Rejected projects Vs Subjects')
plt.xlabel('Subject')
plt.ylabel('Count')
plt.show()
del histData
del acceptedValues
del rejectedValues


# In[33]:


plt.figure(figsize=(18,7))
histData=train_data['project_title'].value_counts().iloc[0:11]
acceptedValues = train_data.where(train_data['project_is_approved']==1)['project_title'].value_counts().iloc[0:11].values
rejectedValues = train_data.where(train_data['project_is_approved']==0)['project_title'].value_counts().iloc[0:11].values
plt.bar(histData.index,acceptedValues)
plt.bar(histData.index,rejectedValues,bottom=acceptedValues)
plt.title('Accepted & Rejected projects Vs Titles')
plt.xlabel('Title')
plt.ylabel('Count')
plt.show()
del histData


# In[34]:


plt.figure(figsize=(18,7))
histData=train_data['teacher_number_of_previously_posted_projects'].value_counts()

acceptedData=train_data.where(train_data['project_is_approved']==1)['teacher_number_of_previously_posted_projects'].value_counts()
rejectedData=train_data.where(train_data['project_is_approved']==0)['teacher_number_of_previously_posted_projects'].value_counts()
sortedAccepted=acceptedData.sort_index()
sortedRejected=rejectedData.sort_index()
sortedData=histData.sort_index()

for index in sortedData.index:
    
    if index not in sortedAccepted.index:
        sortedAccepted[float(index)]=0
    
    if index not in sortedRejected.index:
        sortedRejected[float(index)]=0


plt.bar(sortedAccepted.iloc[0:30].index,sortedAccepted.iloc[0:30].values)
plt.bar(sortedRejected.iloc[0:30].index,sortedRejected.iloc[0:30].values,bottom=sortedAccepted.iloc[0:30].values)
plt.title('Accepted & Rejected projects Vs previously posted projects')
plt.xlabel('No. of previously posted project')
plt.ylabel('Count')
plt.show()
del histData
del sortedAccepted
del sortedRejected
del sortedData


# - We can see that even teachers with 0 previously posted projects have high acceptance rate.

# ### Time series analysis

# In[3]:


from datetime import datetime
def dateStrToMonth(series):
    monthsArr=[]
    for e in series:
        eDate=datetime.strptime(e, '%Y-%m-%d %H:%M:%S')
        monthsArr.append(eDate.month)
    return pd.Series(monthsArr)
    


# In[36]:


dateAccepted_str=train_data.where(train_data['project_is_approved']==1)['project_submitted_datetime'].dropna()
dateRejected_str=train_data.where(train_data['project_is_approved']==0)['project_submitted_datetime'].dropna()
dateArrRejected=dateStrToMonth(dateRejected_str)
dateArrAccepted=dateStrToMonth(dateAccepted_str)


# In[37]:


acceptedMonths=dateArrAccepted.value_counts(sort=False).sort_index()
rejectedMonths=dateArrRejected.value_counts(sort=False).sort_index()

plt.bar(acceptedMonths.index,acceptedMonths.values)
plt.bar(rejectedMonths.index,rejectedMonths.values,bottom=acceptedMonths.values)
plt.title('Accepted & Rejected projects Vs Months')
plt.xlabel('Months')
plt.ylabel('Count')
plt.show()
del dateArrRejected
del dateArrAccepted
del acceptedMonths
del rejectedMonths


# - August month has the second number of proposals followed by September month .

# In[4]:


def dateStrToWeekDay(series):
    daysArr=[]
    for e in series:
        eDate=datetime.strptime(e, '%Y-%m-%d %H:%M:%S')
        daysArr.append(eDate.weekday())
    return pd.Series(daysArr)


# In[39]:


dayArrRejected=dateStrToWeekDay(dateRejected_str)
dayArrAccepted=dateStrToWeekDay(dateAccepted_str)
acceptedDays=dayArrAccepted.value_counts(sort=False).sort_index()
rejectedDays=dayArrRejected.value_counts(sort=False).sort_index()

plt.bar(acceptedDays.index,acceptedDays.values)
plt.bar(rejectedDays.index,rejectedDays.values,bottom=acceptedDays.values)
plt.title('Accepted & Rejected projects Vs Days')
plt.xlabel('Days')
plt.ylabel('Count')
plt.show()
del dayArrRejected
del dayArrAccepted
del acceptedDays
del rejectedDays


# - The number of proposals decreases as we move towards the end of the week.

# In[5]:


def strToDateSeries(series):
    daysArr=[]
    for e in series:
        eDate=datetime.strptime(e, '%Y-%m-%d %H:%M:%S')
        daysArr.append(eDate.date())
    return pd.Series(daysArr)


# In[41]:


dateArrRejected=strToDateSeries(dateRejected_str)
dateArrAccepted=strToDateSeries(dateAccepted_str)
acceptedDates=dateArrAccepted.value_counts(sort=False).sort_index()
rejectedDates=dateArrRejected.value_counts(sort=False).sort_index()

plt.figure(figsize=(16,7))
plt.bar(acceptedDates.index,acceptedDates.values)
plt.bar(rejectedDates.index,rejectedDates.values,bottom=acceptedDates.values)
plt.title('Accepted & Rejected projects Vs Days')
plt.xlabel('Days')
plt.ylabel('Count')
plt.show()
del acceptedValues
del rejectedValues
del dateArrRejected
del dateArrAccepted
del acceptedDates
del rejectedDates
del dateAccepted_str
del dateRejected_str


# - Looks like we have approximately one years' worth of data (May 2016 to April 2017) given in the training set.
# - There is a sudden spike on a single day (Sep 1, 2016) with respect to the number of proposals (may be some specific reason?)
# 
# 

# # Feature Engineering

# #### We will create 4 features from project_submitted_datetime:
# - year
# - month
# - weekday
# - hour

# In[6]:


def dateStrToYear(series):
    daysArr=[]
    for e in series:
        eDate=datetime.strptime(e, '%Y-%m-%d %H:%M:%S')
        daysArr.append(eDate.year)
    return pd.Series(daysArr)


# In[7]:


def dateStrToHour(series):
    daysArr=[]
    for e in series:
        eDate=datetime.strptime(e, '%Y-%m-%d %H:%M:%S')
        daysArr.append(eDate.hour)
    return pd.Series(daysArr)


# In[ ]:


train_data['year'] = dateStrToYear(train_data['project_submitted_datetime'])
train_data['month'] = dateStrToMonth(train_data['project_submitted_datetime'])
train_data['weekday'] = dateStrToWeekDay(train_data['project_submitted_datetime'])
train_data['hour'] = dateStrToHour(train_data['project_submitted_datetime'])


# In[49]:


#test_data['year'] = dateStrToYear(test_data['project_submitted_datetime'])
#test_data['month'] = dateStrToMonth(test_data['project_submitted_datetime'])
#test_data['weekday'] = dateStrToWeekDay(test_data['project_submitted_datetime'])
#test_data['hour'] = dateStrToHour(test_data['project_submitted_datetime'])


# #### We will create 12 features from essays 1 & 2, project_title & project_resource_summary:
# - e1_length
# - e2_length
# - project_title_len
# - project_resource_summary_len
# - e1_word_count
# - e2_word_count
# - project_title_word_count
# - project_resource_summary_word_count
# - e1_word_density
# - e2_word_density
# - project_title_word_density
# - project_resource_summary_word_density

# In[ ]:


train_data['e1_length'] = train_data['project_essay_1'].str.len()
train_data['e2_length'] = train_data['project_essay_2'].str.len()
train_data['project_title_len'] = train_data['project_title'].str.len()
train_data['project_resource_summary_len'] = train_data['project_resource_summary'].str.len()
#Here we need to use a function that counts words
train_data['e1_word_count'] = train_data['project_essay_1'].str.split().str.len()
train_data['e2_word_count'] = train_data['project_essay_2'].str.split().str.len()
train_data['project_title_word_count'] = train_data['project_title'].str.split().str.len()
train_data['project_resource_summary_word_count'] = train_data['project_resource_summary'].str.split().str.len()
#add word density for description as well


# In[ ]:


#test_data['e1_length'] = test_data['project_essay_1'].str.len()
#test_data['e2_length'] = test_data['project_essay_2'].str.len()
#test_data['project_title_len'] = test_data['project_title'].str.len()
#test_data['project_resource_summary_len'] = test_data['project_resource_summary'].str.len()
#Here we need to use a function that counts words
#test_data['e1_word_count'] = test_data['project_essay_1'].str.split().str.len()
#test_data['e2_word_count'] = test_data['project_essay_2'].str.split().str.len()
#test_data['project_title_word_count'] = test_data['project_title'].str.split().str.len()
#test_data['project_resource_summary_word_count'] = test_data['project_resource_summary'].str.split().str.len()
#add word density for description as well


# In[8]:


from nltk.corpus import stopwords
import re

#takes string s & array of stopwords that you want to add to the default stop words
def cleanString(s,specialStopWords=[],bivariate=False):
    s=s.lower()
    s=re.sub('[^A-Za-z0-9\s]+', '', s)
    sTokenized= s.split()
    myStopWords = stopwords.words('english')
    for word in specialStopWords:
        myStopWords.append(word)
    if bivariate:
        cleanS = [word for word in sTokenized if word not in myStopWords and word != 'nan']
        s =' '.join(cleanS)
        cleanS = re.findall("[^\s]+\s[^\s]+", s) + re.findall("[^\s]+\s[^\s]+", s[s.find(' '):])
    else:
        cleanS = [word for word in sTokenized if word not in myStopWords]
    return pd.Series(cleanS)
#takes string s & array of stopwords that you want to add to the default stop words
def wordDensity(s,specialStopWords=[],bivariate=False):
    s=cleanString(s,specialStopWords,bivariate)
    histData=s.value_counts()
    wordsNum=histData.values.sum()
    density=histData[0:20].values/wordsNum
    return pd.Series(data=density,index=histData.index[0:20])
#Takes two serieses and returns an array of the common indecies
def getSpecialStopWords(s1,s2,eps=1):
    sWords=[x for x in s1.index if x in s2.index and abs(s1[x]-s2[x])<=eps]
    return sWords


# In[9]:


def topWords(featureSeries, labelSeries, prefix='',eps=1, n=10, bivariate=False):
    #This function returns the top words in featureSeries (both approved & rejected) according to densities
    
    #Get top n words in the approved set
    text1=featureSeries.where(labelSeries==1).str.cat(sep=' ')
    approved=wordDensity(text1,bivariate=bivariate)[0:n]
    
    #Get top n words in the rejected set
    text0=featureSeries.where(labelSeries==0).str.cat(sep=' ')
    rejected=wordDensity(text0,bivariate=bivariate)[0:n]
    
    #Get common words between them
    SSW=getSpecialStopWords(approved,rejected,eps)
    
    #Get n top words from each set
    approved = wordDensity(text1,SSW,bivariate=bivariate)[0:n+10]
    rejected = wordDensity(text0,SSW,bivariate=bivariate)[0:n+10]
    
    #Adding prefix
    approved.index= prefix + approved.index
    rejected.index= prefix + rejected.index
    
    #print(approved.index)
    #print(rejected.index)
    fullList=set(approved.index.append(rejected.index))
    #Concatenate them to get the full list
    return list(fullList)
    


# ### Univariate word density

# # Refactored Code

# In[ ]:


e1_topwords=topWords(train_data['project_essay_1'],train_data['project_is_approved'],prefix='e1_')
e2_topwords=topWords(train_data['project_essay_2'],train_data['project_is_approved'],prefix='e2_')
project_title_topwords=topWords(train_data['project_title'],train_data['project_is_approved'],prefix='project_title_')
project_resource_summary_topwords=topWords(train_data['project_resource_summary'],train_data['project_is_approved'],prefix='project_resource_summary_')


# #### We create a list of all the indices (combine those from each feature)

# In[ ]:


initArray=np.zeros(train_data.shape[0])

#Change it to a list of the indices instead of being a series
wordsFeaturesExtractedFromDensity = project_resource_summary_topwords + project_title_topwords + e1_topwords + e2_topwords
wordsFeaturesExtractedFromDensity


# #### Here we use the list to add these indices to the dataset & initialize them to 0 values so that we can insert values in them later

# In[ ]:


#So here we need to initialize the 80 columns now.
for keyword in wordsFeaturesExtractedFromDensity:
    train_data[keyword]=pd.Series(initArray)


# In[ ]:


#So here we need to initialize the 80 columns now.
#for keyword in wordsFeaturesExtractedFromDensity:
#    test_data[keyword]=pd.Series(initArray)
del initArray


# In[ ]:


train_data.head()


# #### Extracting values for univariate words in Essay 1

# In[ ]:


for index, text in train_data.project_essay_1.items():
    density=wordDensity(text)
    for word, value in density.items():
        word2='e1_'+word
        if word2 in e1_topwords:
            train_data[word2][index] = value


# #### Extracting values for univariate words in Essay 2

# In[ ]:


for index, text in train_data.project_essay_2.items():
    density=wordDensity(text)
    for word, value in density.items():
        word2='e2_'+word
        if word2 in e2_topwords:
            train_data[word2][index] = value


# #### Extracting values for univariate words in Project_title

# In[ ]:


for index, text in train_data.project_title.items():
    density=wordDensity(text)
    for word, value in density.items():
        word2='project_title_'+word
        if word2 in project_title_topwords:
            train_data[word2][index] = value


# #### Extracting values for univariate words in Project_resource_summary

# In[ ]:


for index, text in train_data.project_resource_summary.items():
    density=wordDensity(text)
    for word, value in density.items():
        word2='project_resource_summary_'+word
        if word2 in project_resource_summary_topwords:
            train_data[word2][index] = value


# In[ ]:


#for index, text in test_data.project_essay_1.items():
 #   density=wordDensity(text)
  #  for word, value in density.items():
   #     word2='e1_'+word
    #    if word2 in e1_topwords:
     #       test_data[word2][index] = value


# #### Extracting values for univariate words in Essay 2

# In[51]:


#for index, text in test_data.project_essay_2.items():
 #   density=wordDensity(text)
  #  for word, value in density.items():
   #     word2='e2_'+word
    #    if word2 in e2_topwords:
     #       test_data[word2][index] = value


# #### Extracting values for univariate words in Project_title

# In[41]:


#for index, text in test_data.project_title.items():
 #   density=wordDensity(text)
  #  for word, value in density.items():
   #     word2='project_title_'+word
    #    if word2 in project_title_topwords:
     #       test_data[word2][index] = value


# #### Extracting values for univariate words in Project_resource_summary

# In[ ]:


#for index, text in test_data.project_resource_summary.items():
 #   density=wordDensity(text)
  #  for word, value in density.items():
   #     word2='project_resource_summary_'+word
    #    if word2 in project_resource_summary_topwords:
     #       test_data[word2][index] = value


# ### Bivariate word density

# In[ ]:


e1_topwords=topWords(train_data['project_essay_1'],train_data['project_is_approved'],prefix='e1_',bivariate=True)
e2_topwords=topWords(train_data['project_essay_2'],train_data['project_is_approved'],prefix='e2_',bivariate=True)
project_title_topwords=topWords(train_data['project_title'],train_data['project_is_approved'],prefix='project_title_',bivariate=True)
project_resource_summary_topwords=topWords(train_data['project_resource_summary'],train_data['project_is_approved'],prefix='project_resource_summary_',bivariate=True)


# #### We create a list of all the indices (combine those from each feature)

# In[ ]:


initArray=np.zeros(train_data.shape[0])

#Change it to a list of the indices instead of being a series
wordsFeaturesExtractedFromDensity = project_resource_summary_topwords + project_title_topwords + e1_topwords + e2_topwords
wordsFeaturesExtractedFromDensity


# #### Here we use the list to add these indices to the dataset & initialize them to 0 values so that we can insert values in them later

# In[ ]:


for keyword in wordsFeaturesExtractedFromDensity:
    train_data[keyword]=pd.Series(initArray)


# In[ ]:


#for keyword in wordsFeaturesExtractedFromDensity:
 #   test_data[keyword]=pd.Series(initArray)
del initArray


# #### Extracting values for bivariate words in Essay 1

# In[ ]:


for index, text in train_data.project_essay_1.items():
    density=wordDensity(text,bivariate=True)
    for word, value in density.items():
        word2='e1_'+word
        if word2 in e1_topwords:
            train_data[word2][index] = value


# #### Extracting values for bivariate words in Essay 2

# In[ ]:


for index, text in train_data.project_essay_2.items():
    density=wordDensity(text,bivariate=True)
    for word, value in density.items():
        word2='e2_'+word
        if word2 in e2_topwords:
            train_data[word2][index] = value


# #### Extracting values for bivariate words in Project_title

# In[ ]:


for index, text in train_data.project_title.items():
    density=wordDensity(text,bivariate=True)
    for word, value in density.items():
        word2='project_title_'+word
        if word2 in project_title_topwords:
            train_data[word2][index] = value


# #### Extracting values for bivariate words in Project_resource_summary

# In[ ]:


for index, text in train_data.project_resource_summary.items():
    density=wordDensity(text,bivariate=True)
    for word, value in density.items():
        word2='project_resource_summary_'+word
        if word2 in project_resource_summary_topwords:
            train_data[word2][index] = value


# In[ ]:


#for index, text in test_data.project_essay_1.items():
 #   density=wordDensity(text,bivariate=True)
  #  for word, value in density.items():
   #     word2='e1_'+word
    #    if word2 in e1_topwords:
     #       test_data[word2][index] = value


# #### Extracting values for bivariate words in Essay 2

# In[ ]:


#for index, text in test_data.project_essay_2.items():
 #   density=wordDensity(text,bivariate=True)
  #  for word, value in density.items():
   #     word2='e2_'+word
    #    if word2 in e2_topwords:
     #       test_data[word2][index] = value


# #### Extracting values for bivariate words in Project_title

# In[ ]:


#for index, text in test_data.project_title.items():
 #   density=wordDensity(text,bivariate=True)
  #  for word, value in density.items():
   #     word2='project_title_'+word
    #    if word2 in project_title_topwords:
     #       test_data[word2][index] = value


# #### Extracting values for bivariate words in Project_resource_summary

# In[ ]:


#for index, text in test_data.project_resource_summary.items():
 #   density=wordDensity(text,bivariate=True)
  #  for word, value in density.items():
   #     word2='project_resource_summary_'+word
    #    if word2 in project_resource_summary_topwords:
     #       test_data[word2][index] = value


# In[ ]:


train_data.head()


# #### We will create 6 Categorical features :
# - teacher_prefix
# - teacher_id
# - school_state
# - project_grade_category
# - project_subject_categories
# - project_subject_subcategories

# In[ ]:


#teacher_prefix
train_data=train_data.join(train_data['teacher_prefix'].str.get_dummies())


# In[ ]:


#project_grade_category
train_data=train_data.join(train_data['project_grade_category'].str.get_dummies())


# In[ ]:


#count_maps
school_state_counts=train_data['school_state'].value_counts().to_dict()
project_subject_categories_counts=train_data['project_subject_categories'].value_counts().to_dict()
project_subject_subcategories_counts=train_data['project_subject_subcategories'].value_counts().to_dict()
teacher_id_counts=train_data['teacher_id'].value_counts().to_dict()


# In[ ]:


#For school_state & cats just use the count method
#For the id we will just ignore it.
train_data['school_state']=train_data['school_state'].map(school_state_counts)
train_data['project_subject_categories']=train_data['project_subject_categories'].map(project_subject_categories_counts)
train_data['project_subject_subcategories']=train_data['project_subject_subcategories'].map(project_subject_subcategories_counts)
train_data['teacher_id']=train_data['teacher_id'].map(teacher_id_counts)


# #### Dropping useless features

# In[ ]:


#dropping processed categorical columns
train_data.drop(columns=['teacher_prefix','project_grade_category'],inplace=True)
#dropping essay 3 & 4 as they have more than 96% nans
train_data.drop(columns=['project_essay_1','project_essay_2','project_essay_3','project_essay_4'],inplace=True)
#dropping datetime
train_data.drop(columns=['project_submitted_datetime'],inplace=True)


# #### Adding features from resources_data

# In[ ]:


resources_agg=resources_data.groupby(['id']).sum()
resources_mean=resources_data.groupby(['id']).mean()
resources_count=resources_data.groupby(['id']).count()


# In[ ]:


resources=resources_agg.join(resources_mean,rsuffix='_mean').join(resources_count,rsuffix='_count')
resources.drop(columns=['quantity_count','price_count'],inplace=True)
resources.rename(columns={'description':'resources_count'},inplace=True)


# In[ ]:


train_data=train_data.join(resources,on='id')


# #### We need to one hot encode year, month, weekday & hour

# In[ ]:


train_data=train_data.join(pd.get_dummies(train_data['year']) ,rsuffix='_year').join(pd.get_dummies(train_data['month']),rsuffix='_month').join(pd.get_dummies(train_data['weekday']),rsuffix='_weekday').join(pd.get_dummies(train_data['hour']),rsuffix='_hour')


# #### We need to drop id, project_title, year, month, weekday, hour & summary

# In[ ]:


train_data.drop(columns=['year','month','weekday','hour','id','project_title','project_resource_summary'],inplace=True)


# In[ ]:


#teacher_prefix
#test_data=test_data.join(test_data['teacher_prefix'].str.get_dummies())


# In[ ]:


#project_grade_category
#test_data=test_data.join(test_data['project_grade_category'].str.get_dummies())


# In[ ]:


#count_maps
#school_state_counts=test_data['school_state'].value_counts().to_dict()
#project_subject_categories_counts=test_data['project_subject_categories'].value_counts().to_dict()
#project_subject_subcategories_counts=test_data['project_subject_subcategories'].value_counts().to_dict()
#teacher_id_counts=test_data['teacher_id'].value_counts().to_dict()


# In[ ]:


#For school_state & cats just use the count method
#For the id we will just ignore it.
#test_data['school_state']=test_data['school_state'].map(school_state_counts)
#test_data['project_subject_categories']=test_data['project_subject_categories'].map(project_subject_categories_counts)
#test_data['project_subject_subcategories']=test_data['project_subject_subcategories'].map(project_subject_subcategories_counts)
#test_data['teacher_id']=test_data['teacher_id'].map(teacher_id_counts)


# #### Dropping useless features

# In[ ]:


#dropping processed categorical columns
#test_data.drop(columns=['teacher_prefix','project_grade_category'],inplace=True)
#dropping essay 3 & 4 as they have more than 96% nans
#test_data.drop(columns=['project_essay_1','project_essay_2','project_essay_3','project_essay_4'],inplace=True)
#dropping datetime
#test_data.drop(columns=['project_submitted_datetime'],inplace=True)


# #### Adding features from resources_data

# In[ ]:


#test_data=test_data.join(resources,on='id')


# #### We need to one hot encode year, month, weekday & hour

# In[ ]:


#test_data=test_data.join(pd.get_dummies(test_data['year']) ,rsuffix='_year').join(pd.get_dummies(test_data['month']),rsuffix='_month').join(pd.get_dummies(test_data['weekday']),rsuffix='_weekday').join(pd.get_dummies(test_data['hour']),rsuffix='_hour')


# #### We need to drop id, project_title, year, month, weekday, hour & summary

# In[ ]:


#test_data.drop(columns=['year','month','weekday','hour','id','project_title','project_resource_summary'],inplace=True)


# In[ ]:


#test_data.head()


# In[12]:


from sklearn.linear_model import LogisticRegression
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split


# In[ ]:


#train_data.to_csv('processed_train_data.csv',index=False)
#test_data.to_csv('processed_test_data.csv',index=False)


# In[1]:


#importing the needed libraries
#import numpy as np
#import pandas as pd
#import matplotlib.pyplot as plt
#import gc
#from sklearn.linear_model import LogisticRegression
#import sklearn.metrics as metrics
#from sklearn.model_selection import train_test_split
#from nltk.corpus import stopwords
#import re
#from os import path
#from PIL import Image
#from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
#%matplotlib inline
#processed_data=pd.read_csv('processed_train_data.csv')
#processed_data.head()


# ## Building our baseline

# In[20]:


#Split data to features and labels
features=np.array(train_data['teacher_number_of_previously_posted_projects']).reshape(-1, 1)
labels=np.array(train_data['project_is_approved'])
#X_test=np.array(test_data['teacher_number_of_previously_posted_projects']).reshape(-1, 1)
#y_test=np.array(test_data['project_is_approved'])

X_train, X_test, y_train, y_test = train_test_split( features, labels, test_size=0.2, random_state=42)

#Choose a model
benchmark_model=LogisticRegression(random_state=42, solver='saga')
#train the model using only previously posted projects
benchmark_model.fit(X_train,y_train)
#test the model
y_pred = benchmark_model.predict_proba(X_test)
#print(y_pred.reshape(1, -1)[0][0::2])
#Check results
fpr_bl, tpr_bl, thresholds_bl = metrics.roc_curve(y_test, y_pred.reshape(1, -1)[0][1::2])
#print(fpr)
roc_auc_bl = metrics.auc(fpr_bl, tpr_bl)


# #### Now that we have our baseline, let's build a better model

# In[12]:


import lightgbm as lgb
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
#Split data to features and labels
features=train_data.drop(columns=['project_is_approved'])
labels=np.array(train_data['project_is_approved'])
#X_test=test_data.drop(columns=['project_is_approved'])
#y_test=np.array(test_data['project_is_approved'])

X_train, X_test, y_train, y_test = train_test_split( features, labels, test_size=0.2, random_state=42)


#Constructing the model
gbm = lgb.LGBMClassifier(num_leaves=31,
                        learning_rate=0.029,
                        n_estimators=400,
                        max_depth=5)
#Training the model
gbm.fit(X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric='auc',
        early_stopping_rounds=5)
#Predicting
y_pred = gbm.predict_proba(X_test, num_iteration=gbm.best_iteration_)
y_pred2 = gbm.predict(X_test, num_iteration=gbm.best_iteration_)
#print(y_pred.reshape(1, -1)[0][0::2])
#Evaluate Model
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred.reshape(1, -1)[0][1::2])
#print(fpr)
roc_auc = metrics.auc(fpr, tpr)
print(precision_score(y_test, y_pred2))
print(recall_score(y_test, y_pred2))
print(accuracy_score(y_test, y_pred2))


# In[27]:


plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'lightgbm--AUC = %0.2f' % roc_auc)
plt.plot(fpr_bl, tpr_bl, color='green', label = 'baseline--AUC = %0.2f' % roc_auc_bl)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[40]:


lgb.plot_importance(gbm,max_num_features=15)


# In[17]:


from keras.models import Sequential
from keras.layers import Dense, Activation
model = Sequential()
model.add(Dense(32, input_dim=266))
model.add(Activation('sigmoid'))
model.add(Dense(16))
model.add(Activation('sigmoid'))
model.add(Dense(8))
model.add(Activation('sigmoid'))
model.add(Dense(1))
model.add(Activation('sigmoid'))


# In[18]:


model.compile(optimizer='rmsprop',loss='binary_crossentropy', metrics=['accuracy'])


# In[19]:


model.fit(X_train, np.array(y_train), epochs=10, batch_size=100,verbose=1)


# In[117]:


y_predNN=model.predict_proba(X_test)
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_predNN)
#print(fpr)
print(metrics.auc(fpr, tpr))
#y_predNN


# In[ ]:




