# Adult_Census_Machine_Learning_Project
A project to familiarize myself with various methods of classification employing machine learning algorithms 

The project uses data from the UCI Machine Learning dataset resources, comprising of 1994 census data of whether an individual makes more
or less than $50k per year. Various attributes are analyzed and used in a binary logistic and random forest model to make accurate predictions.

First, a basic exploratory analysis of the data and to determine which attributes or variables would be predictive. For example, we can look at the variables of sex, occupation, relationsihip status, marital status, age, and other variables. 

# Sex
![Alt Test](https://github.com/claytonblythe/Adult_Census_Machine_Learning_Project/blob/master/sex.png)

# Race
![Alt Test](https://github.com/claytonblythe/Adult_Census_Machine_Learning_Project/blob/master/race.png)

# Age
![Alt Test](https://github.com/claytonblythe/Adult_Census_Machine_Learning_Project/blob/master/age.png)

# Marital Status
![Alt Test](https://github.com/claytonblythe/Adult_Census_Machine_Learning_Project/blob/master/marital_status.png)

# Relationship
![Alt Test](https://github.com/claytonblythe/Adult_Census_Machine_Learning_Project/blob/master/relationship.png)

# Workclass 
![Alt Test](https://github.com/claytonblythe/Adult_Census_Machine_Learning_Project/blob/master/workclass.png)


Classification models using  "age"   "workclass"  "education"      "education_num"  "marital_status"
"occupation"     "relationship"   "race"    "sex"   "capital_gain"  "capital_loss"   "hours_per_week" variables are run on the target variable which is one-hot encoded representing whether a person's income is under or over $50k

Here I compared a binary logistic classification model with a Random Forest model, which is a popular off-the-shelf method for classification problems using high dimensional data with categorical variables. The Random Forest essentially internally converts categorical variables to dummy variables for each factor level, so long as you explicitly format them as a factor. 


Models are then compared with ROC curves for overall performance. Here the red is the Random Forest model with AUC: .94 and the binary logistic model with an AUC: .90

# ROC 
![Alt Test](https://github.com/claytonblythe/Adult_Census_Machine_Learning_Project/blob/master/ROC.png)


