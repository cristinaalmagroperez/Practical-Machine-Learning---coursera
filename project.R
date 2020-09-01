library(lattice)
library(ggplot2)
library(caret)
library(randomForest)
library(rpart)
library(rpart.plot)
set.seed(1357)
# load data
trainUrl <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
testUrl <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

trainingset <- read.csv(url(trainUrl), na.strings=c("NA","#DIV/0!",""))
testingset <- read.csv(url(testUrl), na.strings=c("NA","#DIV/0!",""))

# data load and clean up
trainingset <- read.csv("C:/Users/alexf/Desktop/Practical-Machine-Learning---coursera/pml-training.csv", na.strings=c("NA","#DIV/0!", ""))
testingset <- read.csv("C:/Users/alexf/Desktop/Practical-Machine-Learning---coursera/pml-testing.csv", na.strings=c("NA","#DIV/0!", ""))


# Delete columns with all missing values
trainingset<-trainingset[,colSums(is.na(trainingset)) == 0]
testingset <-testingset[,colSums(is.na(testingset)) == 0]

# Delete variables are irrelevant to our current project: user_name, raw_timestamp_part_1, raw_timestamp_part_,2 cvtd_timestamp, new_window, and  num_window (columns 1 to 7). 
trainingset <-trainingset[,-c(1:7)]
testingset <-testingset[,-c(1:7)]

# partition the data so that 75% of the training dataset into training and the remaining 25% to testing
traintrainset <- createDataPartition(y=trainingset$classe, p=0.75, list=FALSE)
TrainTrainingSet <- trainingset[traintrainset, ] 
TestTrainingSet <- trainingset[-traintrainset, ]


# PREDICTION MODEL 1: DECISION TREE
model1 <- rpart(classe ~ ., data=TrainTrainingSet, method="class")
prediction1 <- predict(model1, TestTrainingSet, type = "class")

# Plot the Decision Tree
rpart.plot(model1, main="Classification Tree", extra=102, under=TRUE, faclen=0)

# Test results on our TestTrainingSet data set:
confusionMatrix(prediction1, TestTrainingSet$classe) #CHECK


# PREDICTION MODEL 2: RANDOM FOREST
model2 <- randomForest(classe ~. , data=TrainTrainingSet, method="class")
# Predicting:
prediction2 <- predict(model2, TestTrainingSet, type = "class")
# Test results on TestTrainingSet data set:
confusionMatrix(prediction2, TestTrainingSet$classe)




> file <- "C:/Users/alexf/Desktop/Practical-Machine-Learning---coursera/pml-training.csv"

> data <- read.csv(file)




