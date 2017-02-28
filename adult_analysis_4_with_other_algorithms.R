#adult Statistical Analysis version 4
#name: Clayton Blythe, 2/12/17
#email: blythec1@central.edu
#Citation: Lichman, M. (2013). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.

rm(list=ls())
abs_path = '/users/claytonblythe/Desktop/Mega/master_code/Statistics_Machine_Learning/'
#abs_path = 'C://Users//CB02033/Documents//Learning R//Adult_Census'
setwd(abs_path)
training = tbl_df(read.csv("adult.data", strip.white=TRUE, header=FALSE))
testing = tbl_df(read.csv("adult.test", strip.white=TRUE, header=FALSE))
seed <- 712
#rename the columns/variables appropriately
var_names <- c("age", "workclass", "fnlwgt", "education", "education_num", "marital_status", "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss", "hours_per_week", "native_country", "income")
names(training)[1:15] <- var_names
names(testing)[1:15]  <- var_names

#now clean the training and testing data so that no rows with missing values are included for analysis
training <- training %>% filter(occupation != "?", native_country !="?", workclass !="?") 
testing <- testing %>% filter(occupation != "?", native_country !="?", workclass !="?") %>% slice(2:length(testing$income))
#cleaning the testing data to remove the periods in the income column and also add target column
testing$income <- substr(as.character(testing$income), 1, nchar(as.character(testing$income))-1)
training <- training %>% mutate(target = ifelse(training$income == ">50K", 1, 0))
testing <- testing %>% mutate(target = ifelse(testing$income == ">50K", 1, 0))

#lets check on the frequency of our target variable for both data sets
mean(training$target)
mean(testing$target)
#fixing issue with random forest's handling of new factors in testing data
training <- training %>% mutate(istest = 0)
testing <- testing %>% mutate(istest = 1)
full_set <- rbind(training, testing)
training_new <- filter(full_set, istest == 0)
testing_new <- filter(full_set, istest == 1)

#get data in appropriate format
training_new$age <- as.numeric(training_new$age)
training_new$education_num <- as.numeric(training_new$education_num)
training_new$capital_gain <- as.numeric(training_new$capital_gain)
training_new$capital_loss <- as.numeric(training_new$capital_loss)
training_new$hours_per_week <- as.numeric(training_new$hours_per_week)
training_new$target <- as.factor(training_new$target)
testing_new$age <- as.numeric(testing_new$age)
testing_new$education_num <- as.numeric(testing_new$education_num)
testing_new$capital_gain <- as.numeric(testing_new$capital_gain)
testing_new$capital_loss <- as.numeric(testing_new$capital_loss)
testing_new$hours_per_week <- as.numeric(testing_new$hours_per_week)
testing_new$target <- as.factor(testing_new$target)
#select useful data for other models to train and test on
training_new <- select(training_new, -fnlwgt, -istest, -native_country, -income)
testing_new <- select(testing_new, -fnlwgt, -istest,-native_country, -income)

#Linear Model
fit.lin <- lm(target~., data=training_new)
lin_prediction <- predict.lm(fit.lin, newdata = testing_new)

#do random forest model
set.seed(seed)
time2 <- proc.time()
registerDoSNOW(makeCluster(4, type="SOCK"))
fit <- foreach(ntree = rep(25, 4), .combine = combine, .packages = "randomForest") %dopar% randomForest(target ~., data=training_new, importance=TRUE, ntree=ntree, na.action=na.omit)
time_rf <- proc.time() - time2
Prediction <- predict(fit, testing_new, type="prob")

#lets compare with a binary logistic model
fit.bin <- glm(target ~., data = training_new, family = binomial)
predicted_results <- predict(fit.bin, newdata = testing_new, type="response" )

#GBM Model
set.seed(seed)
fit.gbm <- gbm(target~., data=training_new, distribution="multinomial", n.trees=5000)
set.seed(seed)
gbm_prediction <- data.frame(predict(fit.gbm,testing_new[-13], n.trees=5000, type="response"))

#SVM Model
set.seed(seed)
fit.svm <- svm(target~., data=training_new, probability=TRUE)
svm_prediction <- predict(fit.svm, newdata=testing_new[-13], probability=TRUE)

testing_new <- testing_new %>% mutate(lin_prediction_prob = lin_prediction)
testing_new <- testing_new %>% mutate(rf_prediction_prob = Prediction[,2])
testing_new <- testing_new %>% mutate(bin_prediction_prob = predicted_results)
testing_new <- testing_new %>% mutate(gbm_prediction_prob = gbm_prediction$X1.5000)
testing_new <- testing_new %>% mutate(svm_prediction_prob = attr(svm_prediction,"prob")[,2])

#make the random forest, binary logistic, and gbm roc curves
ROC1 <- roc(testing_new$target, testing_new$rf_prediction_prob, algorithm=2)
ROC2 <- roc(testing_new$target, testing_new$bin_prediction_prob, algorithm=2)
ROC3 <- roc(testing_new$target, testing_new$gbm_prediction_prob, algorithm=2)
ROC4 <- roc(testing_new$target, testing_new$lin_prediction_prob, algorithm=2)
ROC5 <- roc(testing_new$target, testing_new$svm_prediction_prob, algorithm=2)
plot(ROC1, col="red")
lines(ROC2, col="blue")
lines(ROC3, col="green")
lines(ROC4, col="purple")
lines(ROC5, col="yellow")
ROC1$auc
ROC2$auc
ROC3$auc
ROC4$auc
ROC5$auc
glimpse(ROC1)
#look at overall accuracies
testing_new <- testing_new %>% mutate(rf_pred = ifelse(testing_new$rf_prediction_prob >= .5, 1, 0))
testing_new <- testing_new %>% mutate(bin_pred = ifelse(testing_new$bin_prediction_prob >= .5, 1, 0))
testing_new <- testing_new %>% mutate(gbm_pred = ifelse(testing_new$gbm_prediction_prob >= .5, 1, 0))
testing_new <- testing_new %>% mutate(svm_pred = ifelse(testing_new$svm_prediction_prob >= .5, 1, 0))

"Random Forest overall accuracy"
1 - mean(testing_new$rf_pred != testing_new$target)
"Binary Logistic overall accuracy"
1 - mean(testing_new$bin_pred != testing_new$target)
"GBM overall accuracy"
1 - mean(testing_new$gbm_pred != testing_new$target)
"SVM overall accuracy"
1 - mean(testing_new$svm_pred != testing_new$target)

# confidence_intervals_bin <- data.frame(confint(fit.bin))
# idx <- data.frame(exp(coef(summary(fit.bin))[,1]), coef(summary(fit.bin))[,4])
# names(idx) <- c("Odds Ratio", "P.Value")
# idx <- data.frame(idx,confidence_interval_odds_ratio_LL = exp(confidence_intervals_bin[,1]), confidence_interval_odds_ratio_UL = exp(confidence_intervals_bin[,2]))
# names(idx) <- c("Odds.Ratio", "P.Value", "Conf. Int. (2.5%)", "Conf. Int. (97.5%)")
# idx  <- idx[-1,]
# idx  <- idx[-7,]
# idx <- idx[order(idx$P.Value),]
# print.xtable(xtable(idx, digits = -2), type="html", file="fit.bin_summary.html")
# 

#test other algorithms
training_new <- sample_frac(training_new, size = .4)
dataset <- training_new
control <- trainControl(method="repeatedcv", number=10, repeats=3)
metric <- "Accuracy"
preProcess=c("center", "scale")

# # Linear Discriminant Analysis
set.seed(seed)
fit.lda <- train(target~., data=dataset, method="lda", preprProc=preProcess, metric=metric, trControl=control)
# Logistic Regression
set.seed(seed)
fit.glm <- train(target~., data=dataset, method="glm", metric=metric, trControl=control)
# GLMNET
set.seed(seed)
fit.glmnet <- train(target~., data=dataset, method="glmnet", preprProc=preProcess, metric=metric,  trControl=control)
# SVM Radial
set.seed(seed)
fit.svmRadial <- train(target~., data=dataset, method="svmRadial", preprProc=preProcess, metric=metric,  trControl=control, fit=FALSE)
# kNN
set.seed(seed)
fit.knn <- train(target~., data=dataset, method="knn", preprProc=preProcess, metric=metric,  trControl=control)
# Naive Bayes
set.seed(seed)
fit.nb <- train(target~., data=dataset, method="nb", metric=metric, trControl=control)
# CART
set.seed(seed)
fit.cart <- train(target~., data=dataset, method="rpart", metric=metric, trControl=control)
# C5.0
set.seed(seed)
fit.c50 <- train(target~., data=dataset, method="C5.0", metric=metric, trControl=control)
# Bagged CART
set.seed(seed)
fit.treebag <- train(target~., data=dataset, method="treebag", metric=metric, trControl=control)
# Random Forest
set.seed(seed)
fit.rf <- train(target~., data=dataset, method="rf", metric=metric, trControl=control)
# Stochastic Gradient Boosting (Generalized Boosted Modeling)
set.seed(seed)
fit.gbm <- train(target~., data=dataset, method="gbm", metric=metric, trControl=control, verbose=FALSE)

results <- resamples(list( RF = fit.rf, logistic=fit.glm,  svm=fit.svmRadial,  nb=fit.nb, cart=fit.cart, c50=fit.c50, bagging=fit.treebag,  gbm=fit.gbm))



# Table comparison
summary(results)
bwplot(results)
# Dot-plot comparison
dotplot(results)
summary(fit.bin)
