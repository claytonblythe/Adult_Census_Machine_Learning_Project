#adult5.R
#Clayton Blythe
rm(list=ls())
abs_path <- '/users/claytonblythe/Desktop/Mega/master_code/Statistics_Machine_Learning/'
setwd(abs_path)
train <- tbl_df(read.csv("adult.data", strip.white=TRUE, header=FALSE)) 
test  <- tbl_df(read.csv("adult.test", strip.white=TRUE, header=FALSE))[-1,]
x <- c("ggplot2", "dplyr", "animation", "RColorBrewer", "reshape2", "ggthemes", "gganimate", "nnet", "broom")
lapply(x, require, character.only=TRUE)

#rename the columns/variables appropriately
var_names <- c("age", "workclass", "fnlwgt", "education", "education_num", "marital_status", "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss", "hours_per_week", "native_country", "income")
names(train)[1:15] <- var_names
names(test)[1:15]  <- var_names
test$income <- substr(as.character(test$income), 1, nchar(as.character(test$income))-1)
train <- train %>% filter(occupation != "?", native_country !="?", workclass !="?") %>% select(-fnlwgt, -native_country) %>% mutate(target = ifelse(income == ">50K", 1, 0)) %>% select(-income) %>% droplevels()
test <- test %>% filter(occupation != "?", native_country !="?", workclass !="?") %>% select(-fnlwgt, -native_country) %>% mutate(target = ifelse(income == ">50K", 1, 0)) %>% select(-income) %>% droplevels()
mean(train$target)
mean(test$target)

numeric_var <- c("age", "education_num", "capital_gain", "capital_loss", "hours_per_week" )
train[numeric_var] <- lapply(train[numeric_var], as.numeric)
test[numeric_var] <- lapply(test[numeric_var], as.numeric)
train$target <- as.factor(train$target)
test$target <- as.factor(test$target)
# train$education_num_sqr <- train$education_num ^2
# test$education_num_sqr <- test$education_num ^2
# train$hours_per_week_sqr <- train$hours_per_week ^2
# test$hours_per_week_sqr <- test$hours_per_week ^2
# train$capital_gain_sqr <- train$capital_gain ^2
# test$capital_gain_sqr <- test$capital_gain ^2
# train$capital_loss_sqr <- train$capital_loss ^2
# test$capital_loss_sqr <- test$capital_loss ^2
# train$age_sqr <- train$age ^2
# test$age_sqr <- test$age ^2

# data <- bind_rows(train,test)
# ggplot(data, aes(age)) +geom_density( alpha = 0.7, fill="forestgreen") + scale_y_continuous(name = "Density") + ggtitle("") + theme_economist() + theme(plot.title = element_text(size = 19, family = "Tahoma", face = "bold"), text = element_text(size = 17, family = "Tahoma")) + ylab("Density") 
# 
# ggplot(data, aes(education_num)) +geom_density( alpha = 0.7, fill="forestgreen") + scale_y_continuous(name = "Density") + ggtitle("") + theme_economist() + theme(plot.title = element_text(size = 19, family = "Tahoma", face = "bold"), text = element_text(size = 17, family = "Tahoma")) + ylab("Density") + theme(axis.text.x=element_text(angle=60,vjust=.5))
# 
# ggplot(data, aes(hours_per_week)) +geom_density( alpha = 0.7, fill="forestgreen") + scale_y_continuous(name = "Density") + ggtitle("") + theme_economist() + theme(plot.title = element_text(size = 19, family = "Tahoma", face = "bold"), text = element_text(size = 17, family = "Tahoma")) + ylab("Density") + theme(axis.text.x=element_text(angle=60,vjust=.5))


# ggplot(data, aes(education)) + geom_bar() + facet_wrap(~target)
# ggplot(data, aes(age, fill=target)) +geom_density()
# ggplot(data, aes(hours_per_week, fill=target)) +geom_density()
# ggplot(data, aes(occupation)) + geom_bar()
# ggplot(data, aes(workclass)) + geom_bar()
# ggplot(data, aes(relationship)) + geom_bar()
# ggplot(data, aes(sex)) + geom_bar()
# ggplot(data, aes(marital_status)) + geom_bar()
# ggplot(data, aes(education_num)) + geom_bar()
# 



train <- bind_rows(train,test)
seed <-714
#binary logistic model
set.seed(seed)
fit.bin <- glm(target ~., data = train, family = binomial)
fit.bin.0 <- glm(target ~1, data = train, family = binomial)
anova(fit.bin,fit.bin.0,test="F")
coef <-  tidy(fit.bin)
names(coef) <- c("Attribute", "Estimate", "Standard_Error", "Statistic", "P_Value")
scoef$Estimate <- exp(coef$Estimate)
coef <- arrange(coef, desc(Estimate))

#do random forest model
set.seed(seed)
time2 <- proc.time()
registerDoSNOW(makeCluster(3, type="SOCK"))
fit.rf <- foreach(ntree = rep(250, 3), .combine = combine, .packages = "randomForest") %dopar% randomForest(target ~., data=train, importance=TRUE, ntree=ntree, mtry=2,na.action=na.omit)
time_rf <- proc.time() - time2
varImpPlot(fit.rf)

#put in the predictions into vectors
rf.predict <- predict(fit.rf, test, type="prob")[,2]
bin.predict <- predict(fit.bin, newdata = test, type="response")


#put the predictions in
test$rf.predict <- rf.predict
test$bin.predict <- bin.predict

#make the random forest, binary logistic, and gbm roc curves
ROC1 <- roc(test$target, test$rf.predict, algorithm=2)
ROC2 <- roc(test$target, test$bin.predict, algorithm=2)
plot(ROC1, col="red")
lines(ROC2, col="blue")
ROC1$auc
ROC2$auc


