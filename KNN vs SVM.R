# load libraries
library(tidyverse) # for data manipulation

library(caret) # for model training

library(GGally)# for visualizing and exploring relationships between variables

library(class) # For knn()

library(pROC) #Sampling-over and under, ROC and AUC curve


# set a random seed for reproducibility
set.seed(385)

##Employee data load:
employees = read.csv("EmployeeData.csv")

##first glance at the data set structure:
str(employees) 

## check for missing values:
summary(employees) # there are NA's in multiple columns

# check if NA is read as character as opposed to missing values:

unique(employees$EnvironmentSatisfaction) #shows there are NAs (missing values)
unique(employees$JobSatisfaction)         #Shows there are NAs (missing values)

sum(is.na(employees$EnvironmentSatisfaction)) #returns '25' NAs
sum(is.na(employees$JobSatisfaction))         # returns '20' NAs

sapply(employees, function(x) sum(is.na(x))) # count missing values for each character columns

# next we count the total number of rows with missing values:
num_na = employees %>% 
  filter(is.na(NumCompaniesWorked)|
           is.na(TotalWorkingYears)|
           is.na(EnvironmentSatisfaction)|
           is.na(JobSatisfaction)) %>% nrow()

#73 observations have a missing value in total
# this is 1.66% of the data and is negligible 

###Data pre-processing ----

# filter the rows with missing values and create an updated data set:
employees = employees %>% 
  filter(!is.na(NumCompaniesWorked)&
           !is.na(TotalWorkingYears)&
           !is.na(EnvironmentSatisfaction)&
           !is.na(JobSatisfaction))

# check to see if 73 rows were dropped: 4410-73=4337
nrow(employees) # returns 4337

# check for missing values for the entire data set again:
employees %>% summarize(
  across(everything(), 
         function(x) sum(is.na(x))))

## back to exploring the categorical or character variables:
#starting with 'EnvironmentSatisfaction' and 'JobSatisFaction'
unique(employees$EnvironmentSatisfaction)
unique(employees$JobSatisfaction)

# drop EmployeeID and StandarHours
employees = employees %>% dplyr::select(-EmployeeID, -StandardHours)

#transformed variables to numeric and categorical for the purpose of analysis
employees_updated = employees %>% 
  mutate(
    #variables below are converted to numeric values
    Education = as.numeric(Education),
    JobLevel = as.numeric(JobLevel),
    EnvironmentSatisfaction = as.numeric(EnvironmentSatisfaction),
    JobSatisfaction = as.numeric(JobSatisfaction),
    MaritalStatus=factor(MaritalStatus),
    NumCompaniesWorked = as.numeric(NumCompaniesWorked),
    TotalWorkingYears = as.numeric(TotalWorkingYears),
    #response variable is converted to a binary factor for use in train()
    Attrition = factor(Attrition, levels = c("No", "Yes"))
  )
str(employees_updated$Attrition)
### Step 1: Create a train/test split ----
test_idx =
  createDataPartition(
    employees_updated$Attrition,
    p = 0.3,
    list = FALSE
  )

#create test set
test_set = employees_updated[test_idx, ]

#create training set
training_set = employees_updated[-test_idx, ]

### Step 2: Data Exploration ----
training_set |>
  ggpairs(aes(color = Attrition, alpha = 0.5)) 
# pair plot shows imbalanced classes in 'Attrition'

# visualize the quality variable. 
training_set |>
  ggplot(aes(x = Attrition)) + 
  geom_bar()
#The dataset is imbalanced on Attrition
# This will get our model trained mainly on majority "negative" observations.

# We have to down sample the negative class
# create vector of indices to keep

keep_index = c(
  which(training_set$Attrition == "Yes"),
  which(training_set$Attrition == "No") %>% 
    sample(sum(training_set$Attrition == "Yes"))
)

table(training_set$Attrition[keep_index]) # balanced sample confirmed 

### Step 3: Feature Engineering ----

#Scale predictors because distance functions are sensitive to scale
#The target variable, Attrition, doesn't need to be scaled because it is categorical.

# Use information from train set to scale test dataset!

standardizer = preProcess(
  training_set,
  method = c("center", "scale")
)

training_set = predict(standardizer, training_set)

test_set = predict(standardizer, test_set)

### Step 5: Feature & Model Selection using KNN method ----

k = training_set |>
  nrow() |>
  sqrt() |>
  round()
# k =55

# build a KNN model using caret::train()
# use 10 fold cross validation
# search for the optimal "k" from 2 to 70

knn_classifier = train(
  Attrition ~ .,
  data = training_set,
  method = "knn",
  tuneGrid = expand.grid(k = seq(2, 70)),
  trControl = trainControl(
    method = "cv", number = 10, # 10-fold cross validation
    classProbs = TRUE,  # Enable probability predictions
    summaryFunction = twoClassSummary  # Use twoClassSummary to compute AUC
  ),
  metric = "ROC" # "ROC" gives us AUC & silences warning about Accuracy
)

plot(knn_classifier)

knn_classifier$bestTune

knn_classifier$results

knn_classifier$resample

### Step 6: Model Validation ----
# No need as validation was completed during training.

### Step 7: Predictions and Conclusions ----

# ROC and AUC
roc_knn = 
  roc(
    test_set$Attrition,
    predict(knn_classifier, test_set, type = "prob")[["Yes"]]
  )

plot(roc_knn)

roc_knn$auc

# Confusion matrix statistics
confusion_stats = 
  confusionMatrix(
    data = test_set$Attrition, 
    reference = predict(knn_classifier, test_set, type = "raw"),  #set p =0.5 as the cut off threshold
    positive = "Yes"
  )

confusion_stats$table

confusion_stats$byClass[c("Precision", "Recall")]

#Precison = 0.63 means that 37% of the prediction were wrong.
#Recall = 0.655 means that 65.5% of employees were classified correctly (in the test set)

#-------------------------------------------------------------------------------------------------------------


Attrition_idx <- c(
  which(training_set$Attrition == "Yes"),
  which(training_set$Attrition == "No") |>
    sample(489)
)

table(training_set$Attrition[Attrition_idx])


### Step 5: Feature & Model Selection using SVN method ---

tr_control = trainControl( # store since we will reuse
  method = "cv", number = 10, # 10-fold cross validation
  classProbs = TRUE,  
  summaryFunction = twoClassSummary  # Use twoClassSummary to compute AUC
)

# three SVM models with the linear, radial, and polynomial kernels

svm_linear = train(
  Attrition ~.,
  data = training_set,
  method = "svmLinear",
  tuneGrid = expand.grid(C = c(0.01,0.1,1,5,10)),
  trControl = tr_control,
  metric = "ROC" # "ROC" gives us AUC & silences warning about Accuracy
)

plot(svm_linear)
svm_linear$bestTune


svm_radial = train(
  Attrition ~.,
  data = training_set,
  method = "svmRadial",
  tuneGrid = expand.grid(C = c(0.01,0.1,1,5,10), sigma = c(0.5,1,2,3)),
  trControl = tr_control,
  metric = "ROC" # "ROC" gives us AUC & silences warning about Accuracy
)

plot(svm_radial)
svm_radial$bestTune


svm_poly = train(
  Attrition ~.,
  data = training_set,
  method = "svmPoly",
  tuneLength = 4, # will automatically try different parameters with CV
  trControl = tr_control,
  metric = "ROC" # "ROC" gives us AUC & silences warning about Accuracy
)

plot(svm_poly)

svm_poly$bestTune

### Step 6: Model Validation for SVN ----
validation_table = 
  bind_rows(
    list(
      svm_linear$resample |>
        mutate(
          type = "linear",
          mean_auc = mean(ROC)
        ) |>
        cbind(svm_radial$bestTune),
      svm_radial$resample |>
        mutate(
          type = "radial",
          mean_auc = mean(ROC)
        ) |>
        cbind(svm_radial$bestTune),
      svm_poly$resample |>
        mutate(
          type = "polynomial",
          mean_auc = mean(ROC)
        ) |>
        cbind(svm_poly$bestTune)
    )
  )

validation_table |>
  ggplot(aes(x = ROC)) +
  geom_density(aes(fill = type), alpha = 0.5) + 
  geom_vline(aes(xintercept = mean_auc))+
  facet_wrap(~type)

#The SVM model with the Radial kernel gives the best ROC result.

### Step 7: Predictions and Conclusions for SVM----



predsLinear <- predict(svm_linear, test_set, type = "prob")

rocLinear <- roc(
  test_set$Attrition,
  predsLinear$Yes
)

plot(rocLinear)

rocLinear$auc
# 

# get probabilistic predictions on your test set on your chosen model
preds = predict(svm_radial, test_set, type = "prob")

# plot ROC and calculate AUC
roc_radial = roc(
  test_set$Attrition,
  preds$Yes
)

plot(roc_radial)

roc_radial$auc
#---------------------------------

predsPoly <- predict(svm_poly, test_set, type = "prob")

rocPoly <- roc(
  test_set$Attrition,
  predsPoly$Yes
)

plot(rocPoly)

rocPoly$auc


# calculate precision and recall.
# pick threshold with highest average of sensitivity and specificity

confusion <- confusionMatrix(
  test_set$Attrition,
  preds |>
    mutate(
      class = ifelse(Yes >= 0.5, "Yes", "No") |>
        factor()
    ) |>
    select(class) |>
    unlist()
  
)

confusion$byClass[c("Precision", "Recall")]


