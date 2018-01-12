#HANDLING CATEGORICAL VARIABLES
#in order to encode categorical variables, 2 common methods are used - OneHotEncoder() and get_dummies()
import pandas as pd
import numpy as np
#df=pd.read_csv('auto.csv')
#df_origin=pd.get_dummies(df)
#print(df.head())
#print(df_origin.head())
#then we drop one dummy that is implied from the rest of dummies
#df_origin=df_origin.drop('origin_Asia',axis=1)
#or, as a simpler way
#df_origin=pd.get_dummies(df,drop_first=True)

#X=df_origin.drop('mpg',axis=1).values
#y=df_origin['mpg'].values
#from sklearn.model_selection import train_test_split
#from sklearn.linear_model import Ridge
#X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)
#ridge=Ridge(alpha=0.5,normalize=True).fit(X_train,y_train)
#print(ridge.score(X_test,y_test))

# Import pandas
#import pandas as pd
# Read 'gapminder.csv' into a DataFrame: df
#df=pd.read_csv('gapminder.csv')
# Create a boxplot of life expectancy per region
#df.boxplot('life', 'Region', rot=60)
# Show the plot
#plt.show()
# Create dummy variables with drop_first=True: df_region
#df_region = pd.get_dummies(df, drop_first=True)
# Print the new columns of df_region
#print(df_region.columns)
# Import necessary modules
#from sklearn.linear_model import Ridge
#from sklearn.model_selection import cross_val_score
# Instantiate a ridge regressor: ridge
#ridge = Ridge(alpha=0.5,normalize=True)
# Perform 5-fold cross-validation: ridge_cv
#ridge_cv = cross_val_score(ridge,X,y,cv=5)
# Print the cross-validated scores
#print(ridge_cv)

#HANDLING MISSING VALUES
df=pd.read_csv('diabetes.csv')
print(df.head())
X=df.drop('diabetes',axis=1).values
y=df['diabetes'].values
df.insulin.replace(0,np.nan,inplace=True)
df.triceps.replace(0,np.nan,inplace=True)
df.bmi.replace(0,np.nan,inplace=True)
#one way is to drop whole rows where missing values occur. but this may result in severe size reduction of the dataset
#df=df.dropna()
#a better strategy is imputing missing values. it should be an educated guess, but for continuous variables 'mean' is oftentimes the solution
#from sklearn.linear_model import LogisticRegression
#from sklearn.model_selection import train_test_split
#from sklearn.pipeline import Pipeline
#from sklearn.preprocessing import Imputer
#imp=Imputer(missing_values='NaN',strategy='mean',axis=0)
#logreg=LogisticRegression()
#steps=[('imputation',imp),('logistic_regression',logreg)]
#pipeline=Pipeline(steps)
#X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)
#pipeline.fit(X_train,y_train)
#y_pred=pipeline.predict(X_test)
#print(pipeline.score(X_test,y_test))

# Convert '?' to NaN
#df[df == '?'] = np.nan
# Print the number of NaNs
#print(df.isnull().sum())
# Print shape of original DataFrame
#print("Shape of Original DataFrame: {}".format(df.shape))
# Drop missing values and print shape of new DataFrame
#df = df.dropna()
# Print shape of new DataFrame
#print("Shape of DataFrame After Dropping All Rows with Missing Values: {}".format(df.shape))

#SVM example with 'most_frequent' imputation strategy
# Import the Imputer module
#from sklearn.preprocessing import Imputer
#from sklearn.svm import SVC
# Setup the Imputation transformer: imp
#imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
# Instantiate the SVC classifier: clf
#clf = SVC()
# Import necessary modules
#from sklearn.preprocessing import Imputer
#from sklearn.pipeline import Pipeline
#from sklearn.svm import SVC
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import classification_report
# Setup the pipeline steps: steps
#steps = [('imputation', Imputer(missing_values='NaN', strategy='most_frequent', axis=0)),
#        ('SVM', SVC())]
# Create the pipeline: pipeline
#pipeline = Pipeline(steps)
# Create training and test sets
#X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)
# Fit the pipeline to the train set
#pipeline.fit(X_train,y_train)
# Predict the labels of the test set
#y_pred=pipeline.predict(X_test)
# Compute metrics
#print(classification_report(y_test, y_pred))

#NORMALIZING - CENTERING AND SCALING
#standardization - subtract the mean and divide by variance
#subtract the minimum and divide by the range
#normalization to (-1,1)
#from sklearn.preprocessing import scale
#X_scaled=scale(X)

# Import scale
#from sklearn.preprocessing import scale
# Scale the features: X_scaled
#X_scaled = X_scaled=scale(X)
# Print the mean and standard deviation of the unscaled features
#print("Mean of Unscaled Features: {}".format(np.mean(X)))
#print("Standard Deviation of Unscaled Features: {}".format(np.std(X)))
# Print the mean and standard deviation of the scaled features
#print("Mean of Scaled Features: {}".format(np.mean(X_scaled)))
#print("Standard Deviation of Scaled Features: {}".format(np.std(X_scaled)))

#there is also a standard scaler in Pipeline
#from sklearn.preprocessing import StandardScaler
#steps=[('scaler',StandardScaler()), 'knn',KNeighborsClassifier()]
#pipeline=Pipeline(steps)
#X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=21)
#knn_scaled=pipeline.fit(X_train,y_train)
#y_pred=pipeline.predict(X_test)
#print(accuracy_score(y_test,y_pred))

# Import the necessary modules
#from sklearn.preprocessing import StandardScaler
#from sklearn.pipeline import Pipeline
# Setup the pipeline steps: steps
#steps = [('scaler', StandardScaler()),
#        ('knn', KNeighborsClassifier())]
# Create the pipeline: pipeline
#pipeline = Pipeline(steps)
# Create train and test sets
#X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)
# Fit the pipeline to the training set: knn_scaled
#knn_scaled = pipeline.fit(X_train,y_train)
# Instantiate and fit a k-NN classifier to the unscaled data
#knn_unscaled = KNeighborsClassifier().fit(X_train, y_train)
# Compute and print metrics
#print('Accuracy with Scaling: {}'.format(knn_scaled.score(X_test,y_test)))
#print('Accuracy without Scaling: {}'.format(knn_unscaled.score(X_test,y_test)))

#we can also standardize data along with cross-validation
#steps=[('scaler',StandardScaler()), 'knn',KNeighborsClassifier()]
#pipeline=Pipeline(steps)
#parameters={knn__n_neighbors=np.arange(1,50)}
#X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=21)
#cv=GridSearchCV(pipeline,param_grid=parameters)
#cv.fit(X_train,y_train)
#y_pred=cv.predict(X_test)
#print(cv.best_params_)
#print(cv.score(X_test,y_test))
#print(classification_report(X_test,y_test))

# Setup the pipeline
#steps = [('scaler', StandardScaler()),
#         ('SVM', SVC())]
#pipeline = Pipeline(steps)
# Specify the hyperparameter space
#parameters = {'SVM__C':[1, 10, 100],
#              'SVM__gamma':[0.1, 0.01]}
# Create train and test sets
#X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=21)
# Instantiate the GridSearchCV object: cv
#cv=GridSearchCV(pipeline,param_grid=parameters)
# Fit to the training set
#cv.fit(X_train,y_train)
# Predict the labels of the test set: y_pred
#y_pred = y_pred=cv.predict(X_test)
# Compute and print metrics
#print("Accuracy: {}".format(cv.score(X_test, y_test)))
#print(classification_report(y_test, y_pred))
#print("Tuned Model Parameters: {}".format(cv.best_params_))

# Setup the pipeline steps: steps
#steps = [('imputation', Imputer(missing_values='NaN', strategy='mean', axis=0)),
#         ('scaler', StandardScaler()),
#         ('elasticnet', ElasticNet())]
# Create the pipeline: pipeline
#pipeline = Pipeline(steps)
# Specify the hyperparameter space
#parameters = {'elasticnet__l1_ratio':np.linspace(0,1,30)}
# Create train and test sets
#X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.4,random_state=42)
# Create the GridSearchCV object: gm_cv
#gm_cv = GridSearchCV(pipeline,param_grid=parameters)
# Fit to the training set
#gm_cv.fit(X_train,y_train)
# Compute and print the metrics
#r2 = gm_cv.score(X_test, y_test)
#print("Tuned ElasticNet Alpha: {}".format(gm_cv.best_params_))
#print("Tuned ElasticNet R squared: {}".format(r2))
