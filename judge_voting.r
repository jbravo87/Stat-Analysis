# Import necessary libraries
library(plyr)
library(dplyr)
library(ggplot2)

# Upload data to first data frame.
raw_data <- read.csv("https://raw.githubusercontent.com/fivethirtyeight/data/master/tenth-circuit/tenth-circuit.csv", header = TRUE, stringsAsFactors = FALSE)

names(raw_data)
df1 <- data.frame(raw_data$Judge1, raw_data$Vote1)
typeof(df1$raw_data.Vote1)
# above is integer type

colnames(df1) <- c("Judge", "Vote")
df1
df1$Vote <- as.factor(df1$Vote)
typeof(df1$Vote)

# IMPORTANT: did not initially filter for immigraiton cases.
df1$Vote <- factor(df1$Vote, labels = c("Conservative", "Liberal"))
df1$Judge <- as.factor(df1$Judge)

df2 <- df1 %>% group_by(df1$Judge) %>% count("Conservative")
typeof(df2)
# above is a list

colnames(df2) <- c("Judge", "Vote", "Frequency")
plot(df2$Judge, df2$Frequency)
typeof(df2$Frequency)

summarydata1 <- df1 %>%
  group_by(df1$Vote) %>%
  summarise(Count = n())
summarydata1
typeof(summarydata1)
# above is a list

colnames(summarydata1) <- c("Judge", "Count")
# Histogram
hist(summarydata1$Count)

sum(with(df2, y1=="Liberal"))
sum(with(df2, y1=="Conservative"))
boxplot(with(df2, summary(df2$Frequency)))

plot1 <- ggplot(df2, aes(x = df2$Judge, y = df2$Frequency)) + geom_boxplot()
plot1
plot2 <- barplot(df2$Frequency)
plot2

df1 %>% group_by(df1$Vote)

sum(x1$Vote1)

# Store sum of Vote1 in y2
y2 <- as.factor(x1$Vote1)
typeof(y2)
cumulative_sum <- count(y2)

typeof(count(y2))
cumulative_sum$x
# Cumulative count stored in the following variable.
total_votes <- sum(cumulative_sum$freq)
total_votes

# New data frame with just judge and vote.
df3 <- data.frame(df2$Judge, df2$Frequency)
colnames(df3) <- c("Judge", "Frequency")
df3$Judges <- as.factor(df3$Judge)

# BElow didn't deliver desired output.
#df3%Rate <- df3$Frequency/total_votes

# The following lines uses a for loop to calculate the
# conservative voting rate for each judge and stores it
# into a new column in the data frame.
for(i in 1:length(df3$Frequency)) {
  df3$Rate[i] <- df3$Frequency[i]/total_votes
}

# df2 is already filtered for conservative votes
# for (i in 1:length(df)) {
#   print("Judge ", df3$Judge[i], " has a Conservative Count Rate of: ", df3$Rate[i], "\n")
# }
for (i in df3) {
  print(paste("Judge ", df3$Judge, " has a Conservative Vote Rate of: ", df3$Rate))
}

# Quick try at plotting
ggplot(df3, aes(x = Judge)) + geom_boxplot()

## Adjust some graphical parameters.
par(mar = c(6.1, 4.1, 4.1, 4.1), # change the margins
    lwd = 2, # increase the line thickness
    cex.axis = 1.2 # increase default axis label size
)
#df3 %>% ggplot(mapping = aes(x=Judge, y= Rate)) + geom_boxplot()
boxplot(df3$Judge, df3$Rate)
## Draw x-axis without labels.
axis(side = 1, labels = FALSE)

## Draw y-axis.
axis(side = 2,
     ## Rotate labels perpendicular to y-axis.
     las = 2,
     ## Adjust y-axis label positions.
     mgp = c(3, 0.75, 0))

## Draw the x-axis labels.
text(x = 1:length(df3$Judges),
     ## Move labels to just below bottom of chart.
     y = par("usr")[3] - 0.45,
     ## Use names from the data list.
     labels = unique(df3$Judges),
     ## Change the clipping region.
     xpd = NA,
     ## Rotate the labels by 35 degrees.
     srt = 35,
     ## Adjust the labels to almost 100% right-justified.
     adj = 0.965,
     ## Increase label size.
     cex = 1.2)

########################################
# Below is from an earlier attempt.
#
########################################

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

library(ggstatsplot)
ggstatsplot::ggcoefstats(
  x = logit.model,
  statistic = 'z',
  ##exponentiate = true,
  title=title,
  xlab=xlab,
  ylab=ylab
)
x3 <- with( trainingdata_2, summary(y1))

# Another attempt at a bar plot
vote_plot <- ggplot( trainingdata_2, aes( x = y2, y = y1 ) ) + geom_boxplot() + coord_flip()
vote_plot

# Will use same logic as above but with a new data frame
voteplot2 <- ggplot( some_table, aes( x = n, y = y2 ) ) + geom_boxplot()
voteplot2

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
