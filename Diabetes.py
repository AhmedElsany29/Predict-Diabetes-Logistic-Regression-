#!/usr/bin/env python
# coding: utf-8

# ### About Data
# ##### This dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases. The objective of the dataset is to diagnostically predict whether a patient has diabetes, based on certain diagnostic measurements included in the dataset. Several constraints were placed on the selection of these instances from a larger database. In particular, all patients here are females at least 21 years old of Pima Indian heritage. Pregnancies: To express the Number of pregnancies Glucose: To express the Glucose level in blood BloodPressure: To express the Blood pressure measurement SkinThickness: To express the thickness of the skin Insulin: To express the Insulin level in blood BMI: To express the Body mass index DiabetesPedigreeFunction: To express the Diabetes percentage Age: To express the age Outcome: To express the final result 1 is Yes and 0 is No
# 
# - Business Problem :It is desired to develope a machine learning model that can predict whether people have diabetes when their characteristics specified.

# ### Improt Libraries
# 

# In[1]:


import pandas as pd 
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import missingno as msno
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# In[2]:


df=pd.read_csv("diabetes.csv")


# In[3]:


df.head(10)


# In[4]:


df.shape


# In[5]:


df.info()


# In[6]:


df.describe().style.background_gradient(cmap="twilight")


# In[26]:


columns_to_impute = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

# Perform median imputation, replacing 0s with the median of each column (excluding 0s)
for column in columns_to_impute:
    median_value = df[df[column] != 0][column].median()
    df[column] = df[column].replace(0, median_value)

# Display summary statistics again to verify changes
df[columns_to_impute].describe()


# - Glucose, BloodPressure, SkinThickness, Insulin, and BMI now have more plausible minimum values, indicating that the placeholder values of 0 have been replaced with the median of each respective column.
# 
# - The mean and standard deviation for these columns have adjusted accordingly, providing a more accurate representation of the underlying data distribution. With these adjustments, our dataset should be in a better state for analysis and modeling.0

# In[7]:


msno.bar(df)
plt.show()


# In[27]:


df.isna().sum()


# In[8]:


corr_matrix = df.corr()
corr_matrix['Outcome'].sort_values(ascending=False)


# In[12]:


df.duplicated().sum()


# In[13]:


df.nunique()


# In[14]:


df.groupby('Outcome').mean()


# In[15]:


sns.heatmap(df.corr(),cmap='coolwarm', linewidth=4,linecolor='green', square=True,annot=True,fmt='.2f')
plt.title("Visualize the Correlation Map ",color ="b")
plt.show()


# In[16]:


sns.pairplot(df,hue="Outcome")


# In[17]:


df.hist(bins=10, figsize=(20,15), color='blue', alpha=0.6, hatch='X', rwidth=5);


# In[18]:


df.Age.plot(color="purple",kind="hist")
plt.show()


# In[19]:


sns.violinplot ( data= df ["BMI"], color="g", split=False, cut=0, bw=.3, inner="stick", scale="count")
plt.show()


# In[20]:


X = df.iloc[:,:-1] 
y = df.iloc[:,-1] 


# In[21]:


from sklearn.model_selection import train_test_split


# In[22]:


X_train,X_test,y_train,y_test=train_test_split (X,y,test_size=.25,random_state=42)


# ## Feature Scakling

# In[23]:


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ## Model

# In[24]:


best_model = None
best_accuracy = 0
best_difference = float('inf')

models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Support Vector Machine': SVC(),
    'K-Nearest Neighbors': KNeighborsClassifier()
}

# Fit and predict for each model
for name, model in models.items():
    # Fit the model
    model.fit(X_train, y_train)
    
    # Predict on test data
    y_pred = model.predict(X_test)
    
    # Predict on train data
    y_pred_train = model.predict(X_train)

    # Calculate accuracy
    accuracy_test = accuracy_score(y_test, y_pred)
    accuracy_train = accuracy_score(y_train, y_pred_train)

    # Print model scores
    print(f"Model: {name}")
    print(f"{name} Test Accuracy: {accuracy_test:.4f}")
    print(f"{name} Train Accuracy: {accuracy_train:.4f}")

    print("\nCompare the train-set and test-set accuracy\n")
    print("Check for overfitting and underfitting\n")
    print('Training set score: {:.4f}'.format(model.score(X_train, y_train)))
    print('Test set score: {:.4f}\n'.format(model.score(X_test, y_test)))    
    # Check for overfitting
    difference = abs(accuracy_train - accuracy_test)
    print(f"Difference between training and testing accuracy: {difference:.4f}")
    print(100*"*")

    # Update best model if it has the highest testing accuracy and minimal overfitting
    if accuracy_test > best_accuracy and difference < best_difference:
        best_model = model
        best_accuracy = accuracy_test
        best_difference = difference



print(f"Best Model: {best_model}")
print(f"Best Testing Accuracy: {best_accuracy:.4f}")
print(f"Difference between training and testing accuracy for the best model: {best_difference:.4f}")

