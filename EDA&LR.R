# Analyses préliminaires

data(iris)
head(iris)

#Le nombre d'observations et la dimension de la variable d'entrée x

dim(iris)

#La plage de valeurs et la moyenne de chaque xj
summary(iris)

#La prépondérance de chaque classe

iris$classes <- ifelse(iris$Species =="setosa", "class1","class2")
iris$classes <- factor(iris$classes)
counts <- table(iris$classes)
barplot(counts, main="La prépondérance de chaque classe",
        xlab="classes")
#Des graphiques dans l'espace des variables d'entrées prises 2 par 2 avec les classes représentées en
#couleur (choisir deux paires de variables d'entrées seulement et justifier votre choix).

c<-cor(iris[,1:4], method = c("pearson"))
c

#Sepal.Length is highly correlated with Petal.Length and Petal.Width. Petal.Length is highly correlated with Petal.Width. Therefore, 
#we can use Sepal.Length and Sepal.Width as independent variables. The below eigen value 
#vector shows that the dimension of data can be reduce to 2 dimensions.
ev <- eigen(c)
ev

#complete matrix plot of variables
CL <- iris$classes
pairs(iris[,1:4],col = rainbow(2)[CL],oma=c(4,4,6,12))
par(xpd=TRUE)
legend(0.85,0.6, as.vector(unique(CL)),fill = rainbow(2))

#independent variable plot

CL <- iris$classes
pairs(iris[,1:2],col = rainbow(2)[CL],oma=c(4,4,6,12))
par(xpd=TRUE)
legend(0.85,0.6, as.vector(unique(CL)),fill = rainbow(2))

#Régression logistique
#II faut centrer et réduire les variables d'entrée pour faciliter l'initialisation des paramètres w
iris_cent <- sapply(iris[,1:4], function(x) scale(x, scale=TRUE))
head(iris_cent)

#Il faut trouver des valeurs initiales w0 de façon aléatoire selon une loi normale standard
set.seed(1)
W <- as.vector(c(rnorm(length(iris[,1:4])+1)))
W

# Il faut utiliser la fonction optimx avec comme fonction objective votre fonction calculant l'entropie avec weight decay;
#create binary classification
iris$binary.class <- ifelse(iris$Species =="setosa", 1,0)
iris_cent <- cbind(iris_cent, iris[c("binary.class")])
# iris_cent$bias <-1
head(iris_cent)
#adding a column of 1 as bias
X <- data.matrix(iris_cent[,1:4])
X_tild <- X
bias <- c(rep(1,dim(X_tild)[1]))
X_tild <- as.matrix(cbind(X_tild,bias))
head(X_tild)

#sigmoid function
sigmoid <- function(w,x){
  as.vector(1/(1+exp(-1*as.matrix(x)%*%(w))))
}

#to avoid having y_pred==0 or  y_pred==1 which cause log(y_pred)=inf we should clip data
#reference:https://stackoverflow.com/questions/13868963/clip-values-between-a-minimum-and-maximum-allowed-value-in-r
myClip <- function(x, a, b) {
  ifelse(x <= a,  a, ifelse(x >= b, b, x))
}

#cross evtropy error
xentropy <- function(y,y_pred){
  -(y*log(y_pred)+(1-y)*log(1-y_pred))
}

# loss function with cross entorpy and logistic regression and regularization
loss <- function(w,x,y,lambda){
  1/(length(y))*sum(xentropy(y,myClip(sigmoid(w,x), 1e-9, 1-1e-9)))+lambda/(2*length(y))*sum(w^2)
}

options(warn=-1)
library(optimx)
set.seed(1)
W <- as.vector(c(rnorm(length(iris[,1:4])+1)))
opt<- optimx(W,loss, x=X_tild,y=iris_cent[,5:5],lambda=0,  method = c("Nelder-Mead"), control=list(maxit=5000))
opt

#Une fois que les étapes précédentes sont maîtrisées, il faut coder une fonction permettant d'entraîner la régression logistique. 
#Pour éviter d'utiliser de mauvais paramètres de départ w0, relancez l'optimisation nstart = 5 (un argument de la fonction d'entraînement)
#avec à chaque fois des paramètres initiaux différents et retenez la meilleure optimisation.

optmx_function <- function(x,y,lambda) {
  nstart=5
  error=100000
  for (i in 1:nstart) {
    #randomly chose W
    W <- as.vector(c(rnorm(5)))
    opt<- optimx(par=W,fn=loss, x=x,y=y,lambda=lambda,  method = c("Nelder-Mead"), control=list(maxit=5000))
    #select best error
    if (summary(opt, order = value)[1, ]["value"]<error) {
      opt_best <-opt
      
    }
    
  }
  return (opt_best)
}
optmx_function( X_tild, iris_cent[,5:5], 100000)


# Faire une permutation aléatoire des données pour s'assurer que les classes sont bien mélangées;
iris_xy<-cbind(X_tild , binary.class=iris_cent[,5:5])
iris_xy_perm <- iris_xy[sample(nrow(iris_xy)),]
head(iris_xy_perm)

# Soit K = 10 le nombre de sous-groupes pour la validation croisée. Pour une série de valeurs de Lambda, estimer l'erreur de 
#généralisation avec la validation croisée. Utiliser le taux de mauvaises classifications.
cross_valid_lambda <- function(x,y,lambdas,K=10) {
  valid_list <-c()
  train_list <-c()
  
  result_df<- data.frame( lambdas=lambdas)
  
  for (lambda in lambdas){
    valid_temp_list <-c()
    train_temp_list <-c()
    for (i in 0:(K-1)) {
      start_row<- as.integer(i/10*nrow(x)+1)
      last_row <-as.integer((i+1)/10*nrow(x))
      #diving validation and training data
      x_valid <- x[start_row:last_row,]
      y_valid <- y[start_row:last_row]
      x_train <-x[-c(start_row:last_row),]
      y_tarin <- y[-c(start_row:last_row)]
      #estimation of W parameters from optimx and training data for each lambda
      W <-as.vector(coef(optmx_function(x_train, y_tarin, lambda)))
      #calculating prediction labels using W from training 
      y_pred_valid<-sigmoid(W,x_valid)
      #transforming probabilities to 0 or 1
      y_pred_vali_normal <- ifelse(y_pred_valid>0.5, 1,0)
      misMatch_valid <-sum(abs(y_valid-y_pred_vali_normal))
      #creating a list of misclassification list for each fold
      valid_temp_list <- c(valid_temp_list, misMatch_valid/length(y_valid))
      
      #estimating predictions for training data
      y_pred_train<-sigmoid(W,x_train)
      #transforming prediction to class 0 or 1
      y_pred_train_normal <- ifelse(y_pred_train>0.5, 1,0)
      # calculating misclassification
      misMatch_train <-sum(abs(y_tarin-y_pred_train))
      #creating a list of misclassification
      train_temp_list <- c(train_temp_list, misMatch_train/length(y_tarin))
      
    }
    train_list <- c(train_list, mean(train_temp_list))
    valid_list <- c(valid_list, mean(valid_temp_list))
    
  }
  
  result_df[ , ncol(result_df) + 1] <- train_list                  # Append new column
  colnames(result_df)[ncol(result_df)] <- paste0("train_misclassification")
  result_df[ , ncol(result_df) + 1] <- valid_list                  # Append new column
  colnames(result_df)[ncol(result_df)] <- paste0("valid_misclassification")
  return (result_df)
}

# lambdas<-c(0,0.1,1,5,10, 25)
# lambdas<- seq(from =1, to =100000, by =0.05)
lambdas<- 0.01 * 10^(seq(0,8,1))
result_df <-cross_valid_lambda(iris_xy_perm[,1:5], iris_xy_perm[,6:6] ,lambdas)

result_df

#Tester plusieurs valeurs possibles pour (Lambda) jusqu'à ce que les courbes obtenues soient environ cohérentes avec ce que la théorie prédit.
library("ggplot2")
theme_set(theme_minimal())
colors <- c("train_misclassification" = "blue", "valid_misclassification" = "red")
# library("RColorBrewer")
gfg_plot <- ggplot(data=result_df, aes(lambdas)) +  
  geom_line( aes(y = train_misclassification, color = "train_misclassification"), size = 1) +
  geom_line(aes(y = valid_misclassification, color = "valid_misclassification"), size = 1)+ 
  labs(x = "Lambda",
       y = "misclassificatiom rate",
       color = "Legend")+ 
  scale_color_manual(values = colors)
gfg_plot
