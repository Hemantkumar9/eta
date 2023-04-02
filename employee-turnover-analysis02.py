import numpy as np #linear algebra
import pandas as pd #data processing 

#data visualization
import seaborn as sns 
import matplotlib.pyplot as plt

#machine learning techniques
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier


# **Data Prep**
# 
# Now that we're set up, we must prep the data.
# 
# Let's take a quick look at the data.

# In[2]:


df=pd.read_excel('hr_comma_sep.xlsx')
df.head()


# We then check for null values and take a closer look at the columns in our data set. We have two categorical columns, sales and salary. The rest are numeric. 
# 
# I also changed the dataset to show average weekly hours instead of monthly, as hours by week seems a more intuitive measurement. 
# 
# Lastly, I renamed the "sales" column to "department" to better represent the data.

# In[3]:


df=df.rename(columns={'average_montly_hours':'average_weekly_hours','sales':'department'})
df['average_weekly_hours']=df['average_weekly_hours']*12/52
df.info()


# The number of projects and average weekly hours worked features seem related, so I find the correlation coefficient to make sure they are not highly correlated with eachother (in that case, I would have removed a feature). However, a .4172 correlation coefficient does not seem high enough to warrant removal. 

# In[4]:


print (np.corrcoef(df['number_project'], df['average_weekly_hours']))


# We can draw the following insights about the people in the dataset from analyzing the numeric variables:
# * they have a relatively short tenure with the firm (average of 3.5 years, max of 10 years)
# * they are generally more satisfied than not (.61 average satisfaction level)
# * they are generally above average performers (.716 average rating in their last evaluation)
# * 14.46% (approximately 1 in 7) of the people have had work accidents
# 
# One caveat- it's a bit hard to define relative satisfaction level and rating because we don't have a basis for comparison. For all we know, a .71 rating could just as likely correspond to a "bad" employee; maybe employers don't usually rate below that number. 
#  

# In[5]:


df.describe()


# Not too much to glean when we describe the categorical variables, besides that there are 10 departments and 3 salary bands.

# In[6]:


df.describe(include=['O'])


# **Feature Selection**
# 
# 
# 

# In[7]:


df[['Work_accident', 'left']].groupby(['Work_accident'], as_index=False).mean().sort_values(by='left')


# In[8]:


df[['department', 'left']].groupby(['department'], as_index=False).mean().sort_values(by='left', ascending=False)


# In[9]:


df[['salary', 'left']].groupby(['salary'], as_index=False).mean().sort_values(by='left', ascending=False)


# In[10]:


df[['number_project', 'left']].groupby(['number_project'], as_index=False).mean().sort_values(by='number_project')


# In[11]:


df[['time_spend_company', 'left']].groupby(['time_spend_company'], as_index=False).mean().sort_values(by='time_spend_company')


# To aid with selecting numeric features, I use scatterplots.
# 
# 

# In[12]:


leave_sat=df.groupby('satisfaction_level').agg({'left': lambda x: len(x[x==1])})
leave_sat['total']=df.groupby('satisfaction_level').agg({'left': len})
leave_sat['leave_rate']=leave_sat['left']/leave_sat['total']
leave_sat['satisfaction']=df.groupby('satisfaction_level').agg({'satisfaction_level': 'mean'})
g=sns.lmplot('satisfaction', 'leave_rate',data=leave_sat)


# In[13]:


leave_eval=df.groupby('last_evaluation').agg({'left': lambda x: len(x[x==1])})
leave_eval['total']=df.groupby('last_evaluation').agg({'left': len})
leave_eval['leave_rate']=leave_eval['left']/leave_eval['total']
leave_eval['evaluation']=df.groupby('last_evaluation').agg({'last_evaluation': 'mean'})
gr=sns.lmplot('evaluation', 'leave_rate',data=leave_eval,fit_reg=False)


# In[14]:


leave_hours=df.groupby('average_weekly_hours').agg({'left': lambda x: len(x[x==1])})
leave_hours['total']=df.groupby('average_weekly_hours').agg({'left': len})
leave_hours['leave_rate']=leave_hours['left']/leave_hours['total']
leave_hours['weekly_hours']=df.groupby('average_weekly_hours').agg({'average_weekly_hours': 'mean'})
grid=sns.lmplot('weekly_hours', 'leave_rate',data=leave_hours,fit_reg=False)


# I revisit the department feature, as I think there's something more to it. In an effort to find a distinguishing element between each department, I test the relationship between an employee's department and the amount of work they do per week. My hypothesis is that some departments might work their employees harder than others, which then affects a worker's likelihood of leaving.
# 
# Observations:
# 
# * There are very minimal differences in the weekly hours worked across departments
# 
# Conclusions:
# 
# * The department feature does not seem very useful; we will not include this in our model

# In[15]:


df[['department', 'average_weekly_hours']].groupby(['department'], as_index=False).mean().sort_values(by='average_weekly_hours', ascending=False)


# Note that I chose not to analyze the promotions feature because only 319 employees out of 14,999 were promoted. This percentage is too small for the feature to be a meaningful predictor. 

# In[16]:


(df.promotion_last_5years==1).sum()
df=df.drop(['promotion_last_5years'],axis=1)


# **Data Wrangling **
# 
# Now we perform the feature deletion and banding as specified earlier. To summarize:
# 
# * we drop the work_accident, department, and average_weekly_hours features
# * we band the number_project, time_spend_company (years at the firm), and last_evaluation features

# In[17]:


df=df.drop(['Work_accident','department','average_weekly_hours'],axis=1)
df.columns


# In[18]:


#banding number of projects
bins=[0,2,5,10]
names=[1,0,1]
df['abnormal_proj']=pd.cut(df['number_project'],bins,labels=names)
#banding years at the firm
bins2=[0,1,2,3,4,5,6,100]
names2=['1','2','3','4','5','6','7']
df['years_at_company']=pd.cut(df['time_spend_company'],bins2,labels=names2)
#banding last_evaluation
bins3=[0,.6,.8,1]
names3=[1,0,1]
df['abnormal_eval']=pd.cut(df['last_evaluation'],bins3,labels=names3)

df.head()


# In[ ]:


#cleaning up intermediary/unused columns
df=df.drop(['number_project','time_spend_company','last_evaluation'],axis=1)
df.head()


# In[ ]:


#turning all columns into numeric so that modeling algorithms can run
df['salary']=df['salary'].map({'low':0,'medium':1,'high':2}).astype(int)
pd.to_numeric(df['abnormal_proj'], errors='coerce')
pd.to_numeric(df['years_at_company'], errors='coerce')
pd.to_numeric(df['abnormal_eval'], errors='coerce')
df.head()


# **Modeling**
# 
# We've finally reached the stage of training a model and using the model to make predictions. 
# 
# Our first step is to split our dataset into a training set and test set. We use an 80-20 split, as is standard. 

# In[ ]:


#Modeling
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(df,df['left'],test_size=.2)
X_train=X_train.drop('left',axis=1)
X_test=X_test.drop('left',axis=1)
print (X_train.shape, Y_train.shape)
print (X_test.shape, Y_test.shape)


# We now must decide on the modelling algorithms that we want to apply. The goal of our model is to use a set of employee characteristics to label him/her as "left the firm" or "stayed". Essentially, we are looking for supervised learning algorithms that perform "classification" and "regression". The following are a few algorithms that meet these criteria:
# 
# * Logistic Regression
# * KNN or k-Nearest Neighbors
# * Support Vector Machines
# * Naive Bayes classifier
# * Decision Tree
# 
# I ultimately selected these techniques because they are relatively simple to understand.

# Logistic regression is a type of regression for which the dependent variable is categorical, in this case binary. A binary logistic model predicts the likelihood of the dependent variable being one case or another (whether the employee leaves the firm or not) using the independent variables. 

# In[ ]:


#Log reg
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
acc_log


# The coefficients represent the level of impact each feature has on the probability of an employee leaving. For each one-unit increase in a given feature, the log-odds are scaled by that feature's coefficient.
# 
# Taking a look at the coefficients table, we see that working on an excessive number of projects (both too few or too many) and having a stellar/poor evaluation are correlated strongest with quitting the firm. Similarly, higher satisfaction and salary tend to significantly decrease the probability of leaving the firm. 

# In[ ]:


coeff_df = pd.DataFrame(X_train.columns)
coeff_df.columns = ['Feature']
coeff_df["Coefficient"] = pd.Series(logreg.coef_[0])

coeff_df.sort_values(by='Coefficient', ascending=False)


# The k-Nearest Neighbors algorithm is very simple: a given employee is predicted to quit or stay based on whether the majority of the kth most similar employees have quit or stayed. The KNN algorithm scores higher than the logistic regression method. 

# In[ ]:


#KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
acc_knn


# Support vector classification is a technique that involves constructing hyperplanes to separate data points into classes.
# These hyperplanes are constructed such that the distance between the hyperplane and the nearest point of each class is maximized. The SVC algorithm scores lower than the KNN classifier. 

# In[ ]:


#SVM
svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
acc_svc


# The naive Bayes algorithm classifies points by using the Bayes' theorem and calculating the posteriors for each class given a vector of features. In other words, it uses probability distributions to determine whether an employee with a certain set of features is more likely to leave or stay; it then assigns the employee to the more likely class. The naive Bayes classifier scores lower than the KNN algorithm.

# In[ ]:


#NB
gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
acc_gaussian


# The decision tree algorithm classifies points by iteratively partioning the dataset into branches based on features. For classification trees, the ending nodes are called "leaves", in which points are assigned a label. For each level of the tree, the algorithm splits the data using the feature that provides the most optimal split. The decision tree algorithm scores the highest among the five that we used. 

# In[ ]:


#Decision Tree
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
acc_decision_tree


# We print out the decision tree below to see which features were selected to partition the dataset at each level of the tree. Feature X[2] (which corresponds to abnormal_proj) is chosen as the "best" separator, followed by X[3] and X[0], which refer to the satisfaction level and years worked at the company, respectively. I only print out the first 4 levels due to the tree's enormous size.
# 
# The logistic regression coefficients denote how significantly the odds of an employee leaving will change from a one-unit change in the features; since the years worked feature is inherently larger in value than the other features (which are either binary or from 0 to 1), the coefficients are less telling of a feature's predictive power than we might otherwise assume.

# In[ ]:


from sklearn import tree
import graphviz
dot_data=tree.export_graphviz(decision_tree,out_file=None, max_depth=3)
graph=graphviz.Source(dot_data)
graph


# **Model Evaluation**
# 
# We now summarize the models we used and rank them based on their accuracy. The decision trees algorithm scored the highest, predicting 98% of the test set's employees' decisions to leave.

# In[ ]:


models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Naive Bayes', 'Decision Tree'],
    'Score': [acc_svc, acc_knn, acc_log, 
              acc_gaussian, acc_decision_tree]})
models.sort_values(by='Score', ascending=False)


# In[ ]:





# In[ ]:




