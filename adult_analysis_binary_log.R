#adult Statistical Analysis version 2 from scratch;
rm(list=ls())
library(dplyr)
library(readr)
library(ggplot2)
library(randomForest)
library(pROC)
abs_path = '/users/claytonblythe/Desktop/Mega/master_code/Statistics_Machine_Learning/'
setwd(abs_path)

training = tbl_df(read.csv("adult.data", strip.white=TRUE, header=FALSE))
testing = tbl_df(read.csv("adult.test", strip.white=TRUE, header=FALSE))
#renames the columns/variables appropriately
names(training)[1:15] = c("age", "workclass", "fnlwgt", "education", "education_num", "marital_status", "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss", "hours_per_week", "native_country", "income")
names(testing)[1:15] = c("age", "workclass", "fnlwgt", "education", "education_num", "marital_status", "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss", "hours_per_week", "native_country", "income")

#now clean the training and testing data so that no rows with missing values are included for analysis
training <- training %>% filter(occupation != "?", native_country !="?", workclass !="?") 
testing <- testing %>% filter(occupation != "?", native_country !="?", workclass !="?") %>% slice(2:length(testing$income))
#cleaning the testing data to remove the periods in the income column 
testing$income <- substr(as.character(testing$income), 1, nchar(as.character(testing$income))-1)

training <- training %>% mutate(target = ifelse(training$income == ">50K", 1, 0))
testing <- testing %>% mutate(target = ifelse(testing$income == ">50K", 1, 0))

#lets check on the frequency of our target variable
sum(training$target)/length(training$target)  #plotting the cleaned data, ... ahhh much better
sum(testing$target)/length(testing$target)  #plotting the cleaned data, ... ahhh much better

#fixing issue with factors
training <- training %>% mutate(istest = 0)
testing <- testing %>% mutate(istest = 1)
full_set <- rbind(training, testing)
training_new <- filter(full_set, istest == 0)
testing_new <- filter(full_set, istest == 1)


#untestedddd
model <- glm(as.factor(target) ~ age + workclass + education + education_num + marital_status + occupation + race + sex + hours_per_week,family=binomial(link='logit'),data=training_new)
