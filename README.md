# Adult Census Machine Learning Classification
A project to familiarize myself with various methods of classification employing machine learning algorithms 

This research project is to investigate the extent to which various social and demographic indicators can predict binary classification of income (>$50k or <$50k) and to also describe the magnitude of each’s variable’s effect on the probability that a particular individual will make more or less than fifty thousand dollars. The data is collected from 1994 Census Data, obtained through the UCI Machine Learning dataset resource. For initial investigation, an exploratory data analysis was performed to get a sense of the distribution of the data, and various transformations and variable creation methods were used.  The data contains the following attributes that are available for analysis.

age: continuous. 

workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked. 

fnlwgt: continuous. 

education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool. 

education-num: continuous. 

marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse. 
occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces. 

relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried. 

race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black. 

sex: Female, Male. 

capital-gain: continuous. 

capital-loss: continuous. 

hours-per-week: continuous. 

First, a basic exploratory analysis of the data and to determine which attributes or variables would be predictive. For example, we can look at the variables of sex, occupation, relationship status, marital status, age, and other variables. In the past, these variables have been associated with wage premiums and discrimination in much of past literature. To begin, the income attribute is one-hot encoded into a "target variable" that is either 1 (indicating they make >$50k) or 0 (indicating they make <$50k). Bear in mind this is in 1994, so the real value of this income would be higher in today's dollars. Overall, the data includes 45,000 observations, with roughly 25% of the data having a "1" in the target attribute, indicating that the individual makes more than $50k. 

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

# Occupation Attribute
![Alt Test](https://github.com/claytonblythe/Adult_Census_Machine_Learning_Project/blob/master/occupation.png)

# Education Attribute
![Alt Test](https://github.com/claytonblythe/Adult_Census_Machine_Learning_Project/blob/master/education.png)


Classification models using  "age"   "workclass"  "education"      "education_num"  "marital_status"
"occupation"     "relationship"   "race"    "sex"   "capital_gain"  "capital_loss"   "hours_per_week" variables are run on the target variable which is one-hot encoded representing whether a person's income is under or over $50k

Some of the variables that are not statistically significant may be so because of a smaller number of observations for that variable. It will most likely be unable to bring some of the less significant variables to higher levels of significance, due to the unavailability of additional data and the relatively low proportional representation of these categories in the data. Most of the magnitudes of the exponentiated log-odds coefficients make sense in the light of referencing the base factor level from which they refer to. After using these base levels as a reference, much of the confusion from coefficients subsides. 
An overall strength of the model can be determined by a linear hypothesis comparing a restricted model with an unrestricted model. Overall, this initial logistic model is quite significant with an F statistic of over 500 and an incredibly small p value. 



Here I compared a binary logistic classification model with a Random Forest model, which is a popular off-the-shelf method for classification problems using high dimensional data with categorical variables. The Random Forest essentially internally converts categorical variables to dummy variables for each factor level, so long as you explicitly format them as a factor. For the Random Forest model here I used 750 trees and an mtry value of 2, which seemed to give optimal results.

It turns out that the Random Forest model shows substantial improvement over a standard binary logistic model, which is interesting. I plan on running the same model with a larger number of trees to see if the performance improves at all. 

This is what the data looks like after reading it into R and cleaning the data (getting rid of NA's, "?" values, and cleaning the data)

![Alt Test](https://github.com/claytonblythe/Adult_Census_Machine_Learning_Project/blob/master/train_data.png)


# ROC Performance Comparison
Models are then compared with ROC curves for overall performance. Here the red is the Random Forest model with AUC: .94 and the blue curve is a binary logistic model with an AUC: .90

The area under the Receiver Operating Characteristic curve is a good metric to use, as it shows the sensitivity and specificity of models under various probability cutoff thresholds.

![Alt Test](https://github.com/claytonblythe/Adult_Census_Machine_Learning_Project/blob/master/ROC.png)




