
load.image.file <- function(filename) {
        ret <- list()
        f <- file(filename, "rb")
        readBin(f, "integer", n = 1, size = 4, endian = "big")
        ret$n <- readBin(f, "integer", n = 1, size = 4, endian = "big")
        nrow <- readBin(f, "integer", n = 1, size = 4, endian = "big")
        ncol <- readBin(f, "integer", n = 1, size = 4, endian = "big")
        x <- readBin(f, "integer", n = ret$n * nrow * ncol, size = 1,
            signed = F)
        ret$x <- matrix(x, ncol = nrow * ncol, byrow = T)
        close(f)
        ret
}
load.label.file <- function(filename) {
    f = file(filename, "rb")
    readBin(f, "integer", n = 1, size = 4, endian = "big")
    n = readBin(f, "integer", n = 1, size = 4, endian = "big")
    y = readBin(f, "integer", n = n, size = 1, signed = F)
    close(f)
    y
}
main<-setwd("D:/Polytechnique/Etudes/Trimestre Hiver 2022/MTH8304 - Apprentissage non supervisé et séries chrono/R")
dir <- "D:/Polytechnique/Etudes/Trimestre Hiver 2022/MTH8304 - Apprentissage non supervisé et séries chrono/R/Data/MNIST/"

## read test data

mnist.test <- load.image.file(paste(dir, "t10k-images-idx3-ubyte",
        sep = ""))

#names(mnist.test)
#mnist.test$n

mnist.test.lab <- load.label.file(paste(dir, "t10k-labels-idx1-ubyte",
                                        sep = ""))

length(mnist.train.lab)
#mnist.test.lab[1:5]


## plot a digit
library(RColorBrewer)

#i <- 10
#dig <- (matrix(mnist.test$x[i,],28,28))[,28:1]

#image(dig, col=brewer.pal(9,"Greys"))


## read training data

mnist.train <- load.image.file(paste(dir, "train-images-idx3-ubyte",
        sep = ""))

mnist.train.lab <- load.label.file(paste(dir, "train-labels-idx1-ubyte",
                                        sep = ""))


#sampling 10% of training data to decrease running time
set.seed(123)
train.idx <- sample(nrow(mnist.train$x), as.integer(1/10 * nrow(mnist.train$x)))
random_train_x <- mnist.train$x[train.idx,]
random_train_label <- mnist.train.lab[train.idx]

#sampling 20% of training data to decrease running time
set.seed(123)
test.idx <- sample(nrow(mnist.test$x), as.integer(1/5 * nrow(mnist.test$x)))
random_test_x <- mnist.test$x[test.idx,]
random_test_label <- mnist.test.lab[test.idx]

library("devtools")
#install_github('davpinto/fastknn')
library('fastknn')

#cross-validation with K-nn to fine best number of near neighbors
cv.out <- fastknnCV(random_train_x, as.factor(random_train_label), k = 2:10, method = "dist", folds = 10, eval.metric = "overall_error")
cv.out$cv_table
cv.out$best_k


library("ggplot2")
theme_set(theme_minimal())
# library("RColorBrewer")
k_plot <- ggplot(data=cv.out$cv_table, aes(k)) +  
  geom_line( aes(y = mean), color = "blue") +
  xlab('K') +
  ylab('mean error')
k_plot

## Fit KNN

#predicting label for test data using the best K from cross validation

knn_label <- fastknn(random_train_x, as.factor(random_train_label), random_test_x, k = cv.out$best_k)

MC_knn<-table(knn_label$class,random_test_label)
MC_knn
## Evaluate model on test set
sprintf("Accuracy: %.4f", (1 - classLoss(actual = as.factor(random_test_label), predicted = knn_label$class)))

misclass_knn<-(colSums(MC_knn)+rowSums(MC_knn)-2*diag(MC_knn))/(colSums(MC_knn)+rowSums(MC_knn)-diag(MC_knn))
misclass_knn




#RANDOM FOREST
library("ranger")

#changing sampled training dataset to dataframe and adding label to training
smapled_training_df <- as.data.frame(random_train_x)
smapled_training_df$lab <-random_train_label

#changing sampled training dataset to dataframe and adding label to training
smapled_test_df <- as.data.frame(random_test_x)
smapled_test_df$lab <-random_test_label

rf_mnist <- ranger(lab ~ ., data=smapled_training_df, classification = TRUE, seed=1)

pred.mnist.rf <- predict(rf_mnist, data = smapled_test_df)
rf_cf<-table( pred.mnist.rf$predictions,smapled_test_df$lab)
rf_cf
accuracy_rf<- sum(diag(rf_cf)/sum(rf_cf))
accuracy_rf

misclass_rf<-(colSums(rf_cf)+rowSums(rf_cf)-2*diag(rf_cf))/(colSums(rf_cf)+rowSums(rf_cf)-diag(rf_cf))
misclass_rf
#rf_mnist$confusion.matrix.accuracy
#rf_mnist$prediction.error

#BAGGING
bagging_mnist <- ranger(lab ~ ., data=smapled_training_df, mtry=ncol(random_train_x) ,classification = TRUE,seed=1)

#test Bagging

pred.mnist.bag <- predict(bagging_mnist, data = smapled_test_df)
bag_cf<-table(pred.mnist.bag$predictions,smapled_test_df$lab)
bag_cf
accuracy_bag<- sum(diag(bag_cf)/sum(bag_cf))
accuracy_bag

misclass_bagging<-(colSums(bag_cf)+rowSums(bag_cf)-2*diag(bag_cf))/(colSums(bag_cf)+rowSums(bag_cf)-diag(bag_cf))
misclass_bagging

#bagging_mnist$confusion.matrix
#ADABOOST
library("fastAdaboost")
?adaboost
adaboost.func<-function (nitr,training_df,train_label,test_df,test_label){
  prob_df <- data.frame(example=seq(1, length(test_label), by = 1))
 
  for (label in 0:9) {
    training_df <- as.data.frame(training_df)
    #print(nrow(training_df))
    #creating one to rest label for training
    train_label_bin <- ifelse(train_label==label, label,10)
    training_df$lab <-train_label_bin
    training_df$lab <- factor(training_df$lab )
    
    #adaboost fit
    adaboost_mnist <- adaboost(lab ~ ., training_df, nitr)
    
    #creating one to rest label for test
    test_df <- as.data.frame(test_df)
    test_label_bin <- ifelse(test_label==label, label,10)
    test_df$lab <-test_label_bin
    
    pred <- predict( adaboost_mnist,newdata=test_df)
    #print(length(pred$prob[,1]))
    prob_list <- c(pred$prob[,1])
    prob_df[ , ncol(prob_df) + 1] <- prob_list                  # Append new column
    colnames(prob_df)[ncol(prob_df)] <- paste0("class=", label)
  }
  prob_df$perd_class <-  max.col(prob_df[,2:11], 'first')-1
  print(prob_df)
  return(prob_df$perd_class)
  }

#sample train
set.seed(123)
train.idx <- sample(nrow(mnist.train$x), as.integer(1/10 * nrow(mnist.train$x)))
random_train_x <- mnist.train$x[train.idx,]
random_train_label <- mnist.train.lab[train.idx]

#sample test
set.seed(123)
test.idx <- sample(nrow(mnist.test$x), as.integer(1/5 * nrow(mnist.test$x)))
random_test_x <- mnist.test$x[test.idx,]
random_test_label <- mnist.test.lab[test.idx]


#Adaboost with 10 learners
boost_y_pred_10=adaboost.func(10,random_train_x,random_train_label,random_test_x,random_test_label)

#confusion matrix
cf_adaboost_10<-table(boost_y_pred_10,random_test_label)
cf_adaboost_10
#overall accuracy
accuracy_ada_10<- sum(diag(cf_adaboost_10)/sum(cf_adaboost_10))
accuracy_ada_10

#misclassification rate
misclass_ada_10<-(colSums(cf_adaboost_10)+rowSums(cf_adaboost_10)-2*diag(cf_adaboost_10))/(colSums(cf_adaboost_10)+rowSums(cf_adaboost_10)-diag(cf_adaboost_10))
misclass_ada_10


#Adaboost with 20 learners
boost_y_pred_20=adaboost.func(20,random_train_x,random_train_label,random_test_x,random_test_label)

cf_adaboost_20<-table(boost_y_pred_20,random_test_label)
cf_adaboost_20
accuracy_ada_20<- sum(diag(cf_adaboost_20)/sum(cf_adaboost_20))
accuracy_ada_20
misclass_ada_20<-(colSums(cf_adaboost_20)+rowSums(cf_adaboost_20)-2*diag(cf_adaboost_20))/(colSums(cf_adaboost_20)+rowSums(cf_adaboost_20)-diag(cf_adaboost_20))
misclass_ada_20


#Adaboost with 40 learners
boost_y_pred_40=adaboost.func(40,random_train_x,random_train_label,random_test_x,random_test_label)

cf_adaboost_40<-table(boost_y_pred_40,random_test_label)
cf_adaboost_40
accuracy_ada_40<- sum(diag(cf_adaboost_40)/sum(cf_adaboost_40))
accuracy_ada_40
misclass_ada_40<-(colSums(cf_adaboost_40)+rowSums(cf_adaboost_40)-2*diag(cf_adaboost_40))/(colSums(cf_adaboost_40)+rowSums(cf_adaboost_40)-diag(cf_adaboost_40))
misclass_ada_40

#Adaboost with 60 learners
boost_y_pred_60=adaboost.func(60,random_train_x,random_train_label,random_test_x,random_test_label)

cf_adaboost_60<-table(boost_y_pred_60,random_test_label)
cf_adaboost_60
accuracy_ada_60<- sum(diag(cf_adaboost_60)/sum(cf_adaboost_60))
accuracy_ada_60
misclass_ada_60<-(colSums(cf_adaboost_60)+rowSums(cf_adaboost_60)-2*diag(cf_adaboost_60))/(colSums(cf_adaboost_60)+rowSums(cf_adaboost_60)-diag(cf_adaboost_60))
misclass_ada_60

#plot of misclassification rates


#creating error dataframe

error_df<- data.frame( labels=c(0:9))
error_df$KNN <- c(misclass_knn)
error_df$RF <-c(misclass_rf)
error_df$bagging <-c(misclass_bagging)
error_df$ada10 <-c(misclass_ada_10)
error_df$ada20 <-c(misclass_ada_20)
error_df$ada40 <-c(misclass_ada_40)
error_df$ada60 <-c(misclass_ada_60)
error_df

colors <- c("knn" = "blue", "RF" = "red","bagging" = "green", "ada10" = "orange","ada20" = "violet", "ada40" = "yellow","ada60" = "brown"  )
# library("RColorBrewer")
error_plot <- ggplot(data=error_df, aes(labels)) +  
  geom_line( aes(y = KNN, color = "knn"), size = 1) +
  geom_line(aes(y = RF, color = "RF"), size = 1)+ 
  geom_line(aes(y = bagging, color = "bagging"), size = 1)+ 
  geom_line(aes(y = ada10, color = "ada10"), size = 1)+ 
  geom_line(aes(y = ada20, color = "ada20"), size = 1)+ 
  geom_line(aes(y = ada40, color = "ada40"), size = 1)+ 
  geom_line(aes(y = ada60, color = "ada60"), size = 1)+ 
  
  labs(x = "labels",
       y = "misclassificatiom rate",
       color = "Legend")+ 
  scale_color_manual(values = colors)+
   xlim(0, 9)+
  scale_x_continuous(breaks = seq(0, 9, 1))
error_plot


