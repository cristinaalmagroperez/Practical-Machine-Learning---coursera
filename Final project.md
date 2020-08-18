## Machine Learning Course Project Writeup
Cristina Almagro Pérez

8/18/2020

### Approach
The outcome variable is classe. Six male participants aged between 20-28 years were asked to perform one set of 10 repetitions of the Unilateral Dumbbell Biceps Curl in 5 different fashions: - exactly according to the specification (Class A) - throwing the elbows to the front (Class B) - lifting the dumbbell only halfway (Class C) - lowering the dumbbell only halfway (Class D) - throwing the hips to the front (Class E).
Class A corresponds to the specified execution of the exercise, while the other 4 classes correspond to common mistakes. Prediction evaluations will be based on maximizing the accuracy and minimizing the out-of-sample error. All other available variables after cleaning will be used for prediction. Two models will be tested using decision tree and random forest algorithms. The model with the highest accuracy will be chosen as our final model.

### Reproducibility
An overall pseudo-random number generator seed was set at 1357 for all code. In order to reproduce the results below, the same seed should be used. Different packages were downloaded and installed, such as caret package, rattle and e1071.

### Cross-validation

Cross-validation will be performed by subsampling our training data set randomly without replacement into 2 subsamples: myTraining data (60% of the original Training data set) and myTesting data (40%). Our models will be fitted on the myTesting data set, and tested on the Testing data. Once the most accurate model, is chosen (either decision tree or random forest) we will test the final model on the original Testing data set.

### Expected out-of-sample error

The expected out-of-sample error will correspond to the quantity: 1-accuracy in the cross-validation data. Accuracy is the proportion of correct classified observation over the total sample in the subTestingSet data set. Expected accuracy is the expected accuracy in the out-of-sample data set (i.e. original testing data set). Thus, the expected value of the out-of-sample error will correspond to the expected number of missclassified observations/total observations in the Test data set, which is the quantity: 1-accuracy found from the cross-validation data set.


## Code and generated plots

### Packages, library and seed


```R
library(caret)
library(randomForest)
library(rpart)
library(rpart.plot)
library(RColorBrewer)
library(rattle)
set.seed(1357)
```

### Load and read data


```R
trainUrl <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
testUrl <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
training <- read.csv(url(trainUrl), na.strings=c("NA","#DIV/0!",""))
testing <- read.csv(url(testUrl), na.strings=c("NA","#DIV/0!",""))
```

### Partition of the training set


```R
inTrain <- createDataPartition(y=training$classe, p=0.6, list=FALSE)
myTraining <- training[inTrain, ]; myTesting <- training[-inTrain, ]
dim(myTraining); dim(myTesting)
```


<ol class=list-inline>
	<li>11776</li>
	<li>160</li>
</ol>




<ol class=list-inline>
	<li>7846</li>
	<li>160</li>
</ol>



### Cleaning of the data


```R
myDataNZV <- nearZeroVar(myTraining, saveMetrics=TRUE)
myNZVvars <- names(myTraining) %in% c("new_window", "kurtosis_roll_belt", "kurtosis_picth_belt",
                                      "kurtosis_yaw_belt", "skewness_roll_belt", "skewness_roll_belt.1", "skewness_yaw_belt",
                                      "max_yaw_belt", "min_yaw_belt", "amplitude_yaw_belt", "avg_roll_arm", "stddev_roll_arm",
                                      "var_roll_arm", "avg_pitch_arm", "stddev_pitch_arm", "var_pitch_arm", "avg_yaw_arm",
                                      "stddev_yaw_arm", "var_yaw_arm", "kurtosis_roll_arm", "kurtosis_picth_arm",
                                      "kurtosis_yaw_arm", "skewness_roll_arm", "skewness_pitch_arm", "skewness_yaw_arm",
                                      "max_roll_arm", "min_roll_arm", "min_pitch_arm", "amplitude_roll_arm", "amplitude_pitch_arm",
                                      "kurtosis_roll_dumbbell", "kurtosis_picth_dumbbell", "kurtosis_yaw_dumbbell", "skewness_roll_dumbbell",
                                      "skewness_pitch_dumbbell", "skewness_yaw_dumbbell", "max_yaw_dumbbell", "min_yaw_dumbbell",
                                      "amplitude_yaw_dumbbell", "kurtosis_roll_forearm", "kurtosis_picth_forearm", "kurtosis_yaw_forearm",
                                      "skewness_roll_forearm", "skewness_pitch_forearm", "skewness_yaw_forearm", "max_roll_forearm",
                                      "max_yaw_forearm", "min_roll_forearm", "min_yaw_forearm", "amplitude_roll_forearm",
                                      "amplitude_yaw_forearm", "avg_roll_forearm", "stddev_roll_forearm", "var_roll_forearm",
                                      "avg_pitch_forearm", "stddev_pitch_forearm", "var_pitch_forearm", "avg_yaw_forearm",
                                      "stddev_yaw_forearm", "var_yaw_forearm")
myTraining <- myTraining[!myNZVvars]

dim(myTraining)
#Remove first column
myTraining <- myTraining[c(-1)]
#Cleaning Variables with too many NAs. For Variables that have more than a 70% of NA’s will be left out:
trainingV3 <- myTraining #creating another subset to iterate in loop
for(i in 1:length(myTraining)) { #for every column in the training dataset
  if( sum( is.na( myTraining[, i] ) ) /nrow(myTraining) >= .7 ) { #if n?? NAs > 70% of total observations
    for(j in 1:length(trainingV3)) {
      if( length( grep(names(myTraining[i]), names(trainingV3)[j]) ) ==1)  { #if the columns are the same:
        trainingV3 <- trainingV3[ , -j] #Remove that column
      }   
    } 
  }
}
dim(trainingV3)

myTraining <- trainingV3
rm(trainingV3)
```


<ol class=list-inline>
	<li>11776</li>
	<li>100</li>
</ol>




<ol class=list-inline>
	<li>11776</li>
	<li>58</li>
</ol>



### Apply same cleaning transformations to Testing and Mytesting datasets


```R
clean1 <- colnames(myTraining)
clean2 <- colnames(myTraining[, -58]) #already with classe column removed
myTesting <- myTesting[clean1]
testing <- testing[clean2]
dim(myTesting);dim(testing)
```


<ol class=list-inline>
	<li>7846</li>
	<li>58</li>
</ol>




<ol class=list-inline>
	<li>20</li>
	<li>57</li>
</ol>



### Coerce the data into same type for correct function of decision trees and random forest


```R
for (i in 1:length(testing) ) {
  for(j in 1:length(myTraining)) {
    if( length( grep(names(myTraining[i]), names(testing)[j]) ) ==1)  {
      class(testing[j]) <- class(myTraining[i])
    }      
  }      
}
# make sure Coertion really worked
testing <- rbind(myTraining[2, -58] , testing)
testing <- testing[-1,]
```

### Machine learning algorithm 1 : Decision tree


```R
model1 <- rpart(classe ~ ., data=myTraining, method="class")
fancyRpartPlot(model1)
#predicting
predictions1 <- predict(model1, myTesting, type = "class")
confusionMatrix(predictions1, myTesting$classe)
```

    Warning message:
    "labs do not fit even at cex 0.15, there may be some overplotting"


    Confusion Matrix and Statistics
    
              Reference
    Prediction    A    B    C    D    E
             A 2115   58    6    0    0
             B  105 1432  100    0    0
             C   12   28 1128   66    0
             D    0    0   58 1028   94
             E    0    0   76  192 1348
    
    Overall Statistics
                                              
                   Accuracy : 0.8987          
                     95% CI : (0.8918, 0.9053)
        No Information Rate : 0.2845          
        P-Value [Acc > NIR] : < 2.2e-16       
                                              
                      Kappa : 0.8719          
                                              
     Mcnemar's Test P-Value : NA              
    
    Statistics by Class:
    
                         Class: A Class: B Class: C Class: D Class: E
    Sensitivity            0.9476   0.9433   0.8246   0.7994   0.9348
    Specificity            0.9886   0.9676   0.9836   0.9768   0.9582
    Pos Pred Value         0.9706   0.8748   0.9141   0.8712   0.8342
    Neg Pred Value         0.9794   0.9861   0.9637   0.9613   0.9849
    Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
    Detection Rate         0.2696   0.1825   0.1438   0.1310   0.1718
    Detection Prevalence   0.2777   0.2086   0.1573   0.1504   0.2060
    Balanced Accuracy      0.9681   0.9555   0.9041   0.8881   0.9465



![png](output_15_2.png)


### Machine learning algorithm 2: Random Forest


```R
model2 <- randomForest(classe ~. , data=myTraining)
# Predicting:
prediction2 <- predict(model2, myTesting, type = "class")
confusionMatrix(prediction2, myTesting$classe)
```


    Confusion Matrix and Statistics
    
              Reference
    Prediction    A    B    C    D    E
             A 2232    0    0    0    0
             B    0 1518    2    0    0
             C    0    0 1366    0    0
             D    0    0    0 1285    4
             E    0    0    0    1 1438
    
    Overall Statistics
                                              
                   Accuracy : 0.9991          
                     95% CI : (0.9982, 0.9996)
        No Information Rate : 0.2845          
        P-Value [Acc > NIR] : < 2.2e-16       
                                              
                      Kappa : 0.9989          
                                              
     Mcnemar's Test P-Value : NA              
    
    Statistics by Class:
    
                         Class: A Class: B Class: C Class: D Class: E
    Sensitivity            1.0000   1.0000   0.9985   0.9992   0.9972
    Specificity            1.0000   0.9997   1.0000   0.9994   0.9998
    Pos Pred Value         1.0000   0.9987   1.0000   0.9969   0.9993
    Neg Pred Value         1.0000   1.0000   0.9997   0.9998   0.9994
    Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
    Detection Rate         0.2845   0.1935   0.1741   0.1638   0.1833
    Detection Prevalence   0.2845   0.1937   0.1741   0.1643   0.1834
    Balanced Accuracy      1.0000   0.9998   0.9993   0.9993   0.9985


### Random Forest ML algorithm used for prediction
Random forest yielded better results. It will be the one used for prediction.


```R
# predict outcome levels on the original Testing data set using Random Forest algorithm
predictfinal <- predict(model2, testing, type="class")
predictfinal
```


<dl class=dl-horizontal>
	<dt>1</dt>
		<dd>B</dd>
	<dt>2</dt>
		<dd>A</dd>
	<dt>31</dt>
		<dd>B</dd>
	<dt>4</dt>
		<dd>A</dd>
	<dt>5</dt>
		<dd>A</dd>
	<dt>6</dt>
		<dd>E</dd>
	<dt>7</dt>
		<dd>D</dd>
	<dt>8</dt>
		<dd>B</dd>
	<dt>9</dt>
		<dd>A</dd>
	<dt>10</dt>
		<dd>A</dd>
	<dt>11</dt>
		<dd>B</dd>
	<dt>12</dt>
		<dd>C</dd>
	<dt>13</dt>
		<dd>B</dd>
	<dt>14</dt>
		<dd>A</dd>
	<dt>15</dt>
		<dd>E</dd>
	<dt>16</dt>
		<dd>E</dd>
	<dt>17</dt>
		<dd>A</dd>
	<dt>18</dt>
		<dd>B</dd>
	<dt>19</dt>
		<dd>B</dd>
	<dt>20</dt>
		<dd>B</dd>
</dl>

<details>
	<summary style=display:list-item;cursor:pointer>
		<strong>Levels</strong>:
	</summary>
	<ol class=list-inline>
		<li>'A'</li>
		<li>'B'</li>
		<li>'C'</li>
		<li>'D'</li>
		<li>'E'</li>
	</ol>
</details>

