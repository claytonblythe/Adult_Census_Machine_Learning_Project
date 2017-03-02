# Adult Census Machine Learning Project
A project to familiarize myself with various methods of classification employing machine learning algorithms 

The project uses data from the UCI Machine Learning dataset resources, comprising of 1994 census data of whether an individual makes more
or less than $50k per year. Various attributes are analyzed and used in a binary logistic and random forest model to make accurate predictions.

First, a basic exploratory analysis of the data and to determine which attributes or variables would be predictive. For example, we can look at the variables of sex, occupation, relationship status, marital status, age, and other variables. 

Here the income attribute is transmuted into a "target variable" that is either 1 (indicating they make >$50k) or 0 (indicating they make <$50k) Keep in mind this is in 1994, so the real value of this income would be higher in today's dollars. 

Overall, about 25% of our data has a "1" in the target attribute.

# Sex Attribute
![Alt Test](https://github.com/claytonblythe/Adult_Census_Machine_Learning_Project/blob/master/sex.png)

# Race Attribute
![Alt Test](https://github.com/claytonblythe/Adult_Census_Machine_Learning_Project/blob/master/race.png)

# Age Attribute
![Alt Test](https://github.com/claytonblythe/Adult_Census_Machine_Learning_Project/blob/master/age.png)

# Marital Status Attribute
![Alt Test](https://github.com/claytonblythe/Adult_Census_Machine_Learning_Project/blob/master/marital_status.png)

# Relationship Attribute
![Alt Test](https://github.com/claytonblythe/Adult_Census_Machine_Learning_Project/blob/master/relationship.png)

# Workclass Attribute
![Alt Test](https://github.com/claytonblythe/Adult_Census_Machine_Learning_Project/blob/master/workclass.png)


Classification models using  "age"   "workclass"  "education"      "education_num"  "marital_status"
"occupation"     "relationship"   "race"    "sex"   "capital_gain"  "capital_loss"   "hours_per_week" variables are run on the target variable which is one-hot encoded representing whether a person's income is under or over $50k

Here I compared a binary logistic classification model with a Random Forest model, which is a popular off-the-shelf method for classification problems using high dimensional data with categorical variables. The Random Forest essentially internally converts categorical variables to dummy variables for each factor level, so long as you explicitly format them as a factor. For the Random Forest model here I used 750 trees and an mtry value of 2, which seemed to give optimal results.

It turns out that the Random Forest model shows substantial improvement over a standard binary logistic model, which is interesting. I plan on running the same model with a larger number of trees to see if the performance improves at all. 

This is what the data looks like after reading it into R and cleaning the data (getting rid of NA's, "?" values, and cleaning the data)

![Alt Test](https://github.com/claytonblythe/Adult_Census_Machine_Learning_Project/blob/master/train_data.png)


# ROC Performance Comparison
Models are then compared with ROC curves for overall performance. Here the red is the Random Forest model with AUC: .94 and the blue curve is a binary logistic model with an AUC: .90

The area under the Receiver Operating Characteristic curve is a good metric to use, as it shows the sensitivity and specificity of models under various probability cutoff thresholds.

![Alt Test](https://github.com/claytonblythe/Adult_Census_Machine_Learning_Project/blob/master/ROC.png)


