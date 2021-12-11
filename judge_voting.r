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
model2 <- glm( y ~ x1, data = training_data, family = binomial )

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

contrasts(training_data$y)

model

# Going to convert the list of judges from as string or symbolic
# to a new numeric list in order to plot

judges_factor <- as.factor( training_data$x1 )
judges_numeric <- as.numeric( training_data$x1 )
length(judges_numeric)

# The following is how to go about plotting a
# logistical regression curve.

#some_model <- glm( vs ~ hp, data = mtcars, family = binomial )
#newdata <- data.frame( hp = seq(min(mtcars$hp), max(mtcars$hp ), len = 500))

# Establish new data frame
#newdf <- data.frame( judges_numeric = seq(min(judges_numeric), max(judges_numeric), len = 954))
newdf2 <- data.frame( y, judges_numeric )
newdf3 <- data.frame(judges = judges_numeric)

#newdf$votes <- predict(model, newdf, type = "response" )
#newdf2$votes <- predict(model, newdf2, type = "response" )
#newdata$vs <- predict( some_model, newdata, type = "response" )
newdf3$Vote <- predict( model2, newdf3, type = "response" )

# Plotting Logic.
#plot(vs~hp, data = mtcars, col = "steelblue" )
#lines( vs~hp, newdata, lwd = 2)

plot( judges_numeric ~ newdf2$y, data = training_data, col = "steelblue" )
plot( judges_numeric ~ training_data$y )
plot( training_data$y ~ judges_factor)
plot( Vote ~ judges, data = newdf3, col = "steelblue" )
summary(newdf3$judges)
hist( training_data$x1 ~ training_data$y, data = newdf3, col = "steelblue" )
typeof(newdf3$judges)
