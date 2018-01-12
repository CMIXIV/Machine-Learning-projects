#if the target variable is a categorical one - we do classification
#if the target variable is continuous - it's a regression task
from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
iris=datasets.load_iris()
#print(type(iris))
#print(iris.keys())
#print(type(iris.data))
#print(type(iris.target))
#print(iris.data.shape)
#print(iris.target_names)
X=iris.data
y=iris.target
df=pd.DataFrame(X,columns=iris.feature_names)
#print(df.head())

#VISUAL EDA (EXPLORATORY DATA ANALYSIS)
#_ =pd.plotting.scatter_matrix(df,c=y,figsize=[10,10],s=50,marker='D')
#plt.show()

#import seaborn
#plt.figure()
#sns.countplot(x='missile', hue='party', data=df, palette='RdBu')
#plt.xticks([0,1], ['No', 'Yes'])
#plt.show()

#VISUALIZING DATA FROM MNIST (IMAGES RECOGNITION)
# Import necessary modules
#from sklearn import datasets
#import matplotlib.pyplot as plt
# Load the digits dataset: digits
#digits = datasets.load_digits()
# Print the keys and DESCR of the dataset
#print(digits.keys())
#print(digits.DESCR)
# Print the shape of the images and data keys
#print(digits.images.shape)
#print(digits.data.shape)
# Display digit 1010
#plt.imshow(digits.images[1010], cmap=plt.cm.gray_r, interpolation='nearest')
#plt.show()

#HEATMAP OF PAIRWISE CORRELATIONS
#sns.heatmap(df.corr(), square=True, cmap='RdYlGn')

#k-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier
#indicate the number of nearest neighbors we wish to take into account for classification
#knn=KNeighborsClassifier(n_neighbors=6)
#then we fit the model to predict the target
#knn.fit(iris['data'],iris['target'])
#knn requires the data to be either in the form of a numpy array or pandas array
#it also requires the features to be in the continuous, not categorical, format
#and there are no missing values in the data
#after fitting, we are ready to predict the target on unlabeled data
#prediction=knn.predict(X)
#print('Prediction {}'.format(prediction))

# Import KNeighborsClassifier from sklearn.neighbors
#from sklearn.neighbors import KNeighborsClassifier
# Create arrays for the features and the response variable
#y = df['party'].values
#X = df.drop('party', axis=1).values
# Create a k-NN classifier with 6 neighbors
#knn=KNeighborsClassifier(n_neighbors=6)
# Fit the classifier to the data
#knn.fit(X,y)
# Predict the labels for the training data X
#y_pred = knn.predict(X)
# Predict and print the label for the new data point X_new
#new_prediction = knn.predict(X_new)
#print("Prediction: {}".format(new_prediction))

from sklearn.model_selection import train_test_split
#random_state indicates the seed of random selection which allows to reproduce exactly the same selection later
#stratify=y allows to select the target values as they are distributed in the dataset
#X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=21,stratify=y)
#knn=KNeighborsClassifier(n_neighbors=8)
#knn.fit(X_train,y_train)
#y_pred=knn.predict(X_test)
#print('Test set predictions:\n {}'.format(y_pred))
#show the accuracy of prediction on test data
#print(knn.score(X_test,y_test))

#smaller k of neighbors can lead to overfitting
#larger k of neighbors can lead to underfitting
#5 to 15 neighbors show the best results on test data

#example - training on MNIST dataset
# Import necessary modules
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.model_selection import train_test_split
# Create feature and target arrays
#X = digits.data
#y = digits.target
# Split into training and test set
#X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
# Create a k-NN classifier with 7 neighbors: knn
#knn=KNeighborsClassifier(n_neighbors=7)
# Fit the classifier to the training data
#knn.fit(X_train,y_train)
# Print the accuracy
#print(knn.score(X_test, y_test))

#LOOPING THE NUMBER OF NEIGHBORS TO SEE THE BEST NUMBER
# Setup arrays to store train and test accuracies
#neighbors = np.arange(1, 9)
#train_accuracy = np.empty(len(neighbors))
#test_accuracy = np.empty(len(neighbors))
# Loop over different values of k
#for i, k in enumerate(neighbors):
    # Setup a k-NN Classifier with k neighbors: knn
#    knn = KNeighborsClassifier(n_neighbors=k)
    # Fit the classifier to the training data
#    knn.fit(X_train,y_train)
    #Compute accuracy on the training set
#    train_accuracy[i] = knn.score(X_train,y_train)
    #Compute accuracy on the testing set
#    test_accuracy[i] = knn.score(X_test, y_test)
# Generate plot
#plt.title('k-NN: Varying Number of Neighbors')
#plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')
#plt.plot(neighbors, train_accuracy, label = 'Training Accuracy')
#plt.legend()
#plt.xlabel('Number of Neighbors')
#plt.ylabel('Accuracy')
#plt.show()

#REGRESSION
#boston=pd.read_csv('boston.csv')
#print(boston.head())
#first of all we create the features dataset by dropping the traget variable, and a target dataset consisting only of the target
#X=boston.drop('MEDV',axis=1).values
#y=boston['MEDV'].values
#.values attribute returns numpy arrays
#let's take out a single feature that is column number 5 - number of rooms
#X_rooms=X[:,5]
#then we normalize the data
#y=y.reshape(-1,1)
#X_rooms=X_rooms.reshape(-1,1)
#we do a visual correlation check
#plt.scatter(X_rooms,y)
#plt.ylabel('Value of house')
#plt.xlabel('No of rooms')
#plt.show()

#LINEAR REGRESSION
#any linear regression is about finding the a and b in y=ax+b so to minimize the error/loss/cost function
#Ordinary Least Squares (OLS) - minimizes the sum of squares of residuals (vertical distances between the actual plots and the regression line)

# Import numpy and pandas
#import numpy as np
#import pandas as pd
# Read the CSV file into a DataFrame: df
#df = pd.read_csv('gapminder.csv')
# Create arrays for features and target variable
#y = df['life'].values
#X = df['fertility'].values
# Print the dimensions of X and y before reshaping
#print("Dimensions of y before reshaping: {}".format(y.shape))
#print("Dimensions of X before reshaping: {}".format(X.shape))
# Reshape X and y
#y = y.reshape(-1,1)
#X = X.reshape(-1,1)
# Print the dimensions of X and y after reshaping
#print("Dimensions of y after reshaping: {}".format(y.shape))
#print("Dimensions of X after reshaping: {}".format(X.shape))

from sklearn import linear_model
#reg=linear_model.LinearRegression()
#reg.fit(X_rooms,y)
#prediction_space=np.linspace(min(X_rooms), max(X_rooms)).reshape(-1,1)
#plt.scatter(X_rooms,y,color='blue')
#plt.plot(prediction_space,reg.predict(prediction_space),color='black',linewidth=3)
#plt.show()

#X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)
#reg_all=linear_model.LinearRegression()
#reg_all.fit(X_train,y_train)
#y_pred=reg_all.predict(X_test)
#the .score accuracy method in linear regression returns the R2
#print(reg_all.score(X_test,y_test))

# Import LinearRegression
#from sklearn.linear_model import LinearRegression
# Create the regressor: reg
#reg = LinearRegression()
# Create the prediction space
#prediction_space = np.linspace(min(X_fertility), max(X_fertility)).reshape(-1,1)
# Fit the model to the data
#reg.fit(X_fertility,y)
# Compute predictions over the prediction space: y_pred
#y_pred = reg.predict(prediction_space)
# Print R^2
#print(reg.score(X_fertility, y))
# Plot regression line
#plt.plot(prediction_space, y_pred, color='black', linewidth=3)
#plt.show()

#Root Mean Square Error (RMSE)
# Import necessary modules
#from sklearn.linear_model import LinearRegression
#from sklearn.metrics import mean_squared_error
#from sklearn.model_selection import train_test_split
# Create training and test sets
#X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)
# Create the regressor: reg_all
#reg_all = LinearRegression()
# Fit the regressor to the training data
#reg_all.fit(X_train,y_train)
# Predict on the test data: y_pred
#y_pred=reg_all.predict(X_test)
# Compute and print R^2 and RMSE
#print("R^2: {}".format(reg_all.score(X_test, y_test)))
#rmse = np.sqrt(mean_squared_error(y_test,y_pred))
#print("Root Mean Squared Error: {}".format(rmse))

#K-FOLD CROSS-VALIDATION!
#from sklearn.model_selection import cross_val_score
#reg=linear_model.LinearRegression()
#cv_results=cross_val_score(reg,X,y,cv=5)
#print(cv_results)
#print(np.mean(cv_results))

#OUTPUT OF COEFFICIENTS WEIGHTS
#names=boston.drop('MEDV',axis=1).columns
#reg_coef=reg.fit(X,y).coef_
#_ =plt.plot(range(len(names)),reg_coef)
#_ =plt.xticks(range(len(names)),names,rotation=60)
#_ = plt.ylabel('Coefficients')
#plt.show()

# Import the necessary modules
#from sklearn.linear_model import LinearRegression
#from sklearn.model_selection import cross_val_score
# Create a linear regression object: reg
#reg = LinearRegression()
# Compute 5-fold cross-validation scores: cv_scores
#cv_scores = cross_val_score(reg,X,y,cv=5)
# Print the 5-fold cross-validation scores
#print(cv_scores)
#print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores)))

# Import necessary modules
#from sklearn.linear_model import LinearRegression
#from sklearn.model_selection import cross_val_score
# Create a linear regression object: reg
#reg = LinearRegression()
# Perform 3-fold CV
#cvscores_3 = cross_val_score(reg,X,y,cv=3)
#print(np.mean(cvscores_3))
# Perform 10-fold CV
#cvscores_10 = cross_val_score(reg,X,y,cv=10)
#print(np.mean(cvscores_10))
#measure the time that each 3-fold and 10-fold takes
#%timeit cross_val_score(reg, X, y, cv = 3)
#%timeit cross_val_score(reg, X, y, cv = 10)

#REGULARIZED REGRESSION
#regularization penalizes large coefficients on train set that definitely lead to overfitting
#1. Ridge regression - MOST PREFERRED - where loss function is OLS plus sum of squared coefficients multiplied by constant alpha. we then need to choose alpha. it's called hyperparameter tuning
#from sklearn.linear_model import Ridge
#from sklearn.model_selection import train_test_split
#X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)
#ridge=Ridge(alpha=0.1,normalize=True)
#ridge.fit(X_train,y_train)
#ridge_pred=ridge.predict(X_test)
#print(ridge.score(X_test,y_test))

#function to plot R2 and alpha scores for each alpha
#def display_plot(cv_scores, cv_scores_std):
#    fig = plt.figure()
#    ax = fig.add_subplot(1,1,1)
#    ax.plot(alpha_space, cv_scores)
#    std_error = cv_scores_std / np.sqrt(10)
#    ax.fill_between(alpha_space, cv_scores + std_error, cv_scores - std_error, alpha=0.2)
#    ax.set_ylabel('CV Score +/- Std Error')
#    ax.set_xlabel('Alpha')
#    ax.axhline(np.max(cv_scores), linestyle='--', color='.5')
#    ax.set_xlim([alpha_space[0], alpha_space[-1]])
#    ax.set_xscale('log')
#    plt.show()
# Import necessary modules
#from sklearn.linear_model import Ridge
#from sklearn.model_selection import cross_val_score
# Setup the array of alphas and lists to store scores
#alpha_space = np.logspace(-4, 0, 50)
#ridge_scores = []
#ridge_scores_std = []
# Create a ridge regressor: ridge
#ridge = Ridge(normalize=True)
# Compute scores over range of alphas
#for alpha in alpha_space:
    # Specify the alpha value to use: ridge.alpha
#    ridge.alpha = alpha
    # Perform 10-fold CV: ridge_cv_scores
#    ridge_cv_scores = cross_val_score(ridge,X,y,cv=10)
    # Append the mean of ridge_cv_scores to ridge_scores
#    ridge_scores.append(np.mean(ridge_cv_scores))
    # Append the std of ridge_cv_scores to ridge_scores_std
#    ridge_scores_std.append(np.std(ridge_cv_scores))
# Display the plot
#display_plot(ridge_scores, ridge_scores_std)

#2. Lasso regression - OLS plus sum of absolute values of coefficients multiplied by constant alpha
#from sklearn.linear_model import Lasso
#from sklearn.model_selection import train_test_split
#X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)
#lasso=Lasso(alpha=0.1)
#lasso.fit(X_train,y_train)
#lasso_pred=lasso.predict(X_test)
#print(lasso.score(X_test,y_test))

# Import Lasso
#from sklearn.linear_model import Lasso
# Instantiate a lasso regressor: lasso
#lasso = lasso=Lasso(alpha=0.4,normalize=True)
# Fit the regressor to the data
#lasso.fit(X,y)
# Compute and print the coefficients
#lasso_coef = lasso.coef_
#print(lasso_coef)
# Plot the coefficients
#plt.plot(range(len(df_columns)), lasso_coef)
#plt.xticks(range(len(df_columns)), df_columns.values, rotation=60)
#plt.margins(0.02)
#plt.show()

#CLASS IMBALANCE PROBLEM (E.G. SPAM IS ONLY 1% OF ALL EMAILS)
#BINARY CLASSIFICATIONS
#confusion matrix of predictions - true positive+false positive+true negative+false negative
#accuracy of binary predictions - sum of true predictions/sum of all predictions
#precision/positive predicted value/PPV - true positives / all positives
#recall/sensitivity/hit rate/true positive rate/TPR - true positives / true positives+false negatives
#F1 score=2*precision*recall/(precision+recall)
#from sklearn.metrics import classification_report
#from sklearn.metrics import confusion_matrix
#knn=KNeighborsClassifier(n_neighbors=8)
#X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.4,random_state=42)
#knn.fit(X_train,y_train)
#y_pred=knn.predict(X_test)
#print(confusion_matrix(y_test,y_pred))
#print(classification_report(y_test,y_pred))

#LOGISTIC REGRESSION is used in CLASSIFICATION PROBLEMS!
#for binary classification, it yields probabilities of positive outcome. if it's greater than 0.5, it's labeled 1, if smaller - 0
#votes=pd.read_csv('house-votes-84.csv', header=None)
#df[df == '?'] = np.nan
#df = df.dropna()
#print(votes.head())
#X=votes.drop(0,axis=1).values
#y=votes[0].values

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
#logreg=LogisticRegression()
#X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.4,random_state=42)
#logreg.fit(X_train,y_train)
#y_pred=logreg.predict(X_test)
#print(confusion_matrix(y_test, y_pred))
#print(classification_report(y_test, y_pred))

#ROC CURVE - RECEIVER OPERATING CHARACTERISTIC
from sklearn.metrics import roc_curve
#we choose the column of the predicted probabilities being 1
#y_pred_prob=logreg.predict_proba(X_test)[:,1]
#fpr,tpr,thresholds=roc_curve(y_test,y_pred_prob)
#plt.plot([0,1],[0,1],'k--')
#plt.plot(fpr,tpr,label='Logistic Regression')
#plt.xlabel('False Positive Rate')
#plt.ylabel('True Positive Rate')
#plt.title('ROC Curve')
#plt.show()

#AUC - 'AREA UNDER (ROC) CURVE'
from sklearn.metrics import roc_auc_score
#logreg=LogisticRegression()
#X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.4,random_state=42)
#logreg.fit(X_train,y_train)
#y_pred_prob=logreg.predict_proba(X_test)[:,1]
#roc_auc_score(y_test,y_pred_prob)
#AUC FOR CROSS-VALIDATION
from sklearn.model_selection import cross_val_score
#cv_scores=cross_val_score(logreg,X,y,cv=5,scoring='roc_auc')
#print(cv_scores)

# Import necessary modules
#from sklearn.metrics import roc_auc_score
#from sklearn.model_selection import cross_val_score
# Compute predicted probabilities: y_pred_prob
#y_pred_prob=logreg.predict_proba(X_test)[:,1]
# Compute and print AUC score
#print("AUC: {}".format(roc_auc_score(y_test, y_pred_prob)))
# Compute cross-validated AUC scores: cv_auc
#cv_auc = cross_val_score(logreg,X,y,cv=5,scoring='roc_auc')
# Print list of AUC scores
#print("AUC scores computed using 5-fold cross-validation: {}".format(cv_auc))

#HYPERPARAMETER TUNING (e.g. alpha or k)
#the best practice is to try a bunch of different hyperparameter values, fit all of them separately, see how well each performs, and choose the best one
#from sklearn.model_selection import GridSearchCV
#param_grid={'n_neighbors': np.arange(1,50)}
#knn=KNeighborsClassifier()
#knn_cv=GridSearchCV(knn,param_grid,cv=5)
#knn_cv.fit(X,y)
#print(knn_cv.best_params_)
#print(knn_cv.best_score_)

# Import necessary modules
#from sklearn.linear_model import LogisticRegression
#from sklearn.model_selection import GridSearchCV
# Setup the hyperparameter grid
#c_space = np.logspace(-5, 8, 15)
#param_grid = {'C': c_space}
# Instantiate a logistic regression classifier: logreg
#logreg = LogisticRegression()
# Instantiate the GridSearchCV object: logreg_cv
#logreg_cv = GridSearchCV(logreg,param_grid,cv=5)
# Fit it to the data
#logreg_cv.fit(X,y)
# Print the tuned parameters and score
#print("Tuned Logistic Regression Parameters: {}".format(logreg_cv.best_params_))
#print("Best score is {}".format(logreg_cv.best_score_))

#randomized grid search on DecisionTreeClassifier
# Import necessary modules
#from scipy.stats import randint
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.model_selection import RandomizedSearchCV
# Setup the parameters and distributions to sample from: param_dist
#param_dist = {"max_depth": [3, None],
#              "max_features": randint(1, 9),
#              "min_samples_leaf": randint(1, 9),
#              "criterion": ["gini", "entropy"]}
# Instantiate a Decision Tree classifier: tree
#tree = DecisionTreeClassifier()
# Instantiate the RandomizedSearchCV object: tree_cv
#tree_cv = RandomizedSearchCV(tree, param_dist, cv=5)
# Fit it to the data
#tree_cv.fit(X,y)
# Print the tuned parameters and score
#print("Tuned Decision Tree Parameters: {}".format(tree_cv.best_params_))
#print("Best score is {}".format(tree_cv.best_score_))

# Import necessary modules
#from sklearn.model_selection import train_test_split
#from sklearn.linear_model import LogisticRegression
#from sklearn.model_selection import GridSearchCV
# Create the hyperparameter grid
#c_space = np.logspace(-5, 8, 15)
#param_grid = {'C': c_space, 'penalty': ['l1', 'l2']}
# Instantiate the logistic regression classifier: logreg
#logreg = LogisticRegression()
# Create train and test sets
#X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.4,random_state=42)
# Instantiate the GridSearchCV object: logreg_cv
#logreg_cv = GridSearchCV(logreg,param_grid,cv=5)
# Fit it to the training data
#logreg_cv.fit(X,y)
# Print the optimal parameters and best score
#print("Tuned Logistic Regression Parameter: {}".format(logreg_cv.best_params_))
#print("Tuned Logistic Regression Accuracy: {}".format(logreg_cv.best_score_))

#ELASTIC NET
#it uses a combination of level 1 (absolute values) and level 2 (squared values) penalties
# Import necessary modules
#from sklearn.linear_model import ElasticNet
#from sklearn.metrics import mean_squared_error
#from sklearn.model_selection import GridSearchCV
#from sklearn.model_selection import train_test_split
# Create train and test sets
#X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.4,random_state=42)
# Create the hyperparameter grid
#l1_space = np.linspace(0, 1, 30)
#param_grid = {'l1_ratio': l1_space}
# Instantiate the ElasticNet regressor: elastic_net
#elastic_net = ElasticNet()
# Setup the GridSearchCV object: gm_cv
#gm_cv = GridSearchCV(elastic_net,param_grid,cv=5)
# Fit it to the training data
#gm_cv.fit(X_train,y_train)
# Predict on the test set and compute metrics
#y_pred = gm_cv.predict(X_test)
#r2 = gm_cv.score(X_test, y_test)
#mse = mean_squared_error(y_test,y_pred)
#print("Tuned ElasticNet l1 ratio: {}".format(gm_cv.best_params_))
#print("Tuned ElasticNet R squared: {}".format(r2))
#print("Tuned ElasticNet MSE: {}".format(mse))
