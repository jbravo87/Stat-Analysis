# First want to upload a clean up the data a bit.
raw_data <- read.csv("https://raw.githubusercontent.com/fivethirtyeight/data/master/tenth-circuit/tenth-circuit.csv", header = TRUE, stringsAsFactors = FALSE)

# checking the various columns in raw data.
names(raw_data)

# Will fit a logistic regression model in order to predict
# Vote1 using Judge1, Judge2, Judge3, and Category.
# Per the README, 1 = liberal vote, 0 = conservative vote

# First filter will be by immigration in the category column.
x1 <- data.frame( raw_data )
x2 <- x1[x1$Category == "Immigration",]
# The last variable stores just the cases that deal with immigration.

# Next objective is to create a data frame with the three columns of interest.
# Vote1, Judge1, and Category
# For the binary 0/1 (conservative/liberal) dealing with nominal categorical variables.
y1 <- factor(x2$Vote1, labels = c("Conservative", "Liberal"))


y2 <- as.factor(x2$Judge1)

# Following line gives all the names of the judges.
levels(y2)

# Now to unclass and assign each judge theor own respective number.
y3 <- unclass(y2)
print(y3)
typeof(y3)

y4 <- as.factor(y2)
y4
typeof(y4)

# Create a final data frame to create model.
training_data = data.frame(x2$Date, y1, y2)

# Want data frame with just judge and vote
trainingdata_2 <- data.frame( y1, y2 )

# Now to code the model using the two categorical variables.
# logistical regression aka logit
logit.model <- glm( y1 ~ y2, data = training_data, family = binomial )

# summary statistics
summary( logit.model )

# Will access just the coefficients for this fitted model.
coef( logit.model )

# More summary data.
summary(logit.model)$coef
summary(logit.model)$coef[ , 4 ]

# Function to predict the probability that the
# judge will vote liberal given predictor values.
model.probs <- predict(logit.model, type = "response")
model.probs

# To get the different categorical values.
unique(y2)

# GOing to use some libraries for further analysis
library(dplyr)
a1 <- trainingdata_2 %>% count("Liberal")
trainingdata_2 %>% count("Conservative", "Liberal")

# Misc code
sum(with(trainingdata_2, y1=="Liberal"))
sum(with(trainingdata_2, y1=="Conservative"))

# Will create two different data sets for liberal vs conservative votes
conservative_votes <- filter( trainingdata_2, y1 == "Conservative" )

# Need the numeric labels for the second column
judges_numeric <- as.numeric( trainingdata_2$y2 )
# data frame with new two columns
newdf3 <- data.frame(y1, judges_numeric)
summarydata <- newdf3 %>%
  group_by(judges_numeric) %>%
  summarise(COunt = n())

summarydata2 <- conservative_votes %>%
  group_by(y2) %>%
  summarise(Count = n())

typeof(summarydata2)
# The above is a list type of data structure

# Another approach to count number of occurences in a column
newdf4 <- table( conservative_votes$y2 )
# delivers an integer data type though.

some_table <- conservative_votes %>% count(y2)

# Will try and get it in one pipe
some_table %>% pull(n) %>% hist()

# experimenting with some plotting
library(ggplot2)
#qplot( some_table, geom = "histogram" )
# Line above doesn't work
hist(some_table$n)
plot(some_table)

hist(some_table$n, col = "steelblue")

# Give the chart file a name.
png(file = "histogram_lim_breaks.png")

# Attempting some intial plotting.
plot(model.probs)
plot(logit.model)

hist( y1 ~ y2, data = training_data, col = "steelblue" )

cs <- cumsum(training_data$y2)

# Following will give the number of occurences for each event.
z1 = table(training_data$y1)


# judges_factor <- as.factor( training_data$x1 )
# judges_numeric <- as.numeric( training_data$x1 )
# length(judges_numeric)

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
coef( logit.model )

# More summary data.
summary(logit.model)$coef
summary(logit.model)$coef[ , 4 ]

# Function to predict the probability that the
# judge will vote liberal given predictor values.
model.probs <- predict(logit.model, type = "response")
model.probs

# To get the different categorical values.
unique(y2)

# Attempting some intial plotting.
plot(model.probs)
plot(logit.model)

hist( y1 ~ y2, data = training_data, col = "steelblue" )

cs <- cumsum(training_data$y2)

# Following will give the number of occurences for each event.
z1 = table(training_data$y1)


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
