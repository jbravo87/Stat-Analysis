# First want to upload a clean up the data a bit.
raw_data <- read.csv("https://raw.githubusercontent.com/fivethirtyeight/data/master/tenth-circuit/tenth-circuit.csv", header = TRUE, stringsAsFactors = FALSE)

# checking the various columns in raw data.
names(raw_data)

# Will fit a logistic regression model in order to predict
# Vote1 using Judge1, Judge2, Judge3, and Category.
# Per the README, 1 = liberal vote, 0 = conservative vote

y <- as.factor(raw_data$Vote1)
x1 <- as.factor(raw_data$Judge1)
x2 <- as.factor(raw_data$Judge2)
x3 <- as.factor(raw_data$Judge3)
x4 <- as.factor(raw_data$Category)
training_data = data.frame(y, x1, x4)
  
cor(training_data)

model <- glm( y ~ x1 + x4, data = training_data, family = binomial )

# summary statistics
summary( model )

# Will access just the coefficients for this fitted model.
coef( model )

summary(model)$coef
summary(model)$coef[ , 4 ]

# Predict function to predict the probability that the
# judge will vote liberal given predictor values.

model.probs <- predict(model, type = "response")
model.probs[ 1:10 ]

contrasts(vote1)
