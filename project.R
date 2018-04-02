library(caret)

train_data_orig <- read.table('/Users/Anne/Documents/*Work/Coursera/Course 8 - Machine Learning/pml-training.csv',sep=',',header=T)
test_data_orig <- read.table('/Users/Anne/Documents/*Work/Coursera/Course 8 - Machine Learning/pml-testing.csv',sep=',',header=T)

#find columns that are mostly NA and factors (except for the classe column)
na_cols <- which(apply(train_data_orig,2,function(x){length(which(is.na(x)))})>(0.5*ncol(train_data_orig)))

train_data <- train_data_orig[,-c(1:7,na_cols)]
test_data <- test_data_orig[,-c(1:7,na_cols)]

cl <- c()
for (i in 1:85){
  cl <- c(cl,class(train_data[,i]))
} 
factor_cols <- which(cl=="factor")

train_data <- train_data[,-factor_cols]
test_data <- test_data[,-factor_cols]


# break up the data into training and evaluation partitions, so we can  evaluate our model
inTrain <- createDataPartition(y=train_data$classe, p=0.60, list=FALSE)
train  <- train_data[inTrain,]
eval  <- train_data[-inTrain,]


# Preprocess using principal components
preProc <- preProcess(train[,-53],method="pca",pcaComp=10)
trainPC <- predict(preProc,train[,-53])
evalPC <- predict(preProc,eval[,-53])
testPC <- predict(preProc,test_data[,-53])

# Build the model
mod <- randomForest(train$classe~.,data=trainPC,ntree=50)
pred <- predict(mod,evalPC)

confusionMatrix(eval$classe,pred)

# Predict for test data
new_data <- predict(mod,testPC)


