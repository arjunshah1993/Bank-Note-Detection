# Imports
library(caTools)
library(neuralnet)
set.seed(101)

# Data import, check its head, structure and check for null values
dataset = read.csv('bank_note_data.csv')
print(head(dataset))
print(str(dataset))
print(anyNA(dataset))

# Split data into training and test sets
split = sample.split(dataset$Class, SplitRatio = 0.7)
training_set = subset(dataset, split == T)
test_set = subset(dataset, split == F)

# Set up the neural net formula for model
n = names(dataset)
formula = as.formula(paste("Class ~", paste(n[!n %in% "Class"], collapse = '+')))

# Train the model with 10 neurons in 1 hidden layer and check the architecture
nn = neuralnet(formula, data = training_set, linear.output = F, hidden = 10)
print(plot(nn))

# Predict results for the test set
y_pred = compute(nn, test_set[, 1:4])

# Make a data frame for the predictions and convert to class from probabilities
predictions = data.frame(Predictions = y_pred$net.result)
predictions = sapply(predictions$Predictions, round)

# Print the confusion matrix
print(table(predictions, test_set$Class))

# We receive a 100% accuracy. Let us compare this to a random forest model
library(randomForest)

# Do a train test split 
split = sample.split(dataset, SplitRatio = 0.7)
training_set2 = subset(dataset, split == T)
test_set2 = subset(dataset, split == F)

# Convert the Class column to factor for random forest model to work
training_set2$Class = factor(training_set2$Class)
test_set2$Class = factor(test_set2$Class)

# Train the random forest model for classification
forest = randomForest(Class ~ ., data = training_set2)

# Get predictions for the test data
y_pred2 = predict(forest, newdata = test_set2[, 1:4])

# Print the confusion matrix
print(table(y_pred2, test_set2$Class))

# Both the models perform exceptionally well and achieve near perfect accuracy