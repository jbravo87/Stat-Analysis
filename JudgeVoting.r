######################################################################
# First part will focus on exploring some simple questions.
# 1. With what frequency did each respective judge vote conservatively
#    on immigration cases in the first vote (Vote1)?
# Will implemeny descriptive statistic
######################################################################

# Import necessary libraries
#library(plyr)
library(dplyr)
library(ggplot2)

# Upload data to first data frame.
# Character vectors will be converted to factors.
raw_data <- read.csv("https://raw.githubusercontent.com/fivethirtyeight/data/master/tenth-circuit/tenth-circuit.csv", header = TRUE, stringsAsFactors = TRUE)

# Test the data type for Vote1 column in dataset
# Check type
# Check type of data
# Check (data) type
typeof(raw_data$Vote1[])
# Still reads as integer
# Since it wasn't a string so read 0 and 1 as integers.
typeof(raw_data$Judge1)
# Above still reads as an integer.

class(raw_data$Judge1)
# The above gives factor.

str(raw_data$Judge1)
# Also gives a factor data type
# Check type complete.

# Name check
names(raw_data)

# First data frame filters for immigration cases, judges, and first vote.
df1 <- data.frame(raw_data$Judge1, raw_data$Vote1, raw_data$Category)
colnames(df1) <- c("Judge", "Vote", "Category")
# Notice use of filter function
df1 <- filter(df1, Category == "Immigration")
# Convert vote to factor
df1$Vote <- factor(df1$Vote, labels = c("Conservative", "Liberal"))

# Will store conservative vote counts in own variable.
##cons_vote <- df1 %>% group_by(df1$Judge) %>% count("Conservative") %>% arrange(desc())
cons_vote <- df1 %>% group_by(df1$Judge) %>% count("Conservative") %>% arrange(desc(n))
# For line above, kind of need to know third column aka count is titled 'n'

# Type check
str(cons_vote)
class(cons_vote)
# S3 dataframe/table
colnames(cons_vote) <- c("Judge", "Vote", "Frequency")
cons_vote <- cons_vote %>%
  arrange(desc(Frequency))
# Above explicitely 

hist(cons_vote$Frequency)
plot(cons_vote$Judge, cons_vote$Frequency)

#ggplot(cons_vote, aes(x = Frequency, y=Judge)) + geom_col()
#ggplot(cons_vote, aes(x = cons_vote$Frequency)) + geom_bar(stat = "identity")
# Some problems with line above.

# vjust = 1.5 <- to get within 
# ggplot(cons_vote, aes(x = reorder(Judge, -Frequency), y = Frequency)) + 
#   geom_bar(stat = "identity") + 
#   geom_text(aes(label = Frequency), vjust = 0.000150, colour = "RED") +
#   theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))

ggplot(cons_vote, aes(x = reorder(Judge, -Frequency), y = Frequency)) + 
  geom_bar(stat = "identity") + 
  geom_text(aes(label = Frequency), vjust = 1.5, colour = "RED") +
  theme(axis.text.x = element_text(angle = 45, vjust = 1.0, hjust=1)) +
  xlab("10th Circuit Judges")

top5_conservative <- head(cons_vote)
# Below is stratch work.
class(top5_conservative)
boxplot(top5_conservative)
# End of scratchwork

# Now want to store cumulative sum in a variable
by_vote <- df1 %>%
  group_by(Vote) %>%
  count()
total_votes <- sum(by_vote$n)
# Type Check!
#str(cumulative_sum)
str(total_votes)
# Indeed an imteger

# Initialize empty column to soon store probabilities.
cons_vote["Probability"] <- NA

# The following lines uses a for loop to calculate the
# conservative voting rate for each judge and stores it
# into a new column in the data frame.
for(i in 1:length(cons_vote$Frequency)) {
  cons_vote$Probability[i] <- cons_vote$Frequency[i]/total_votes
}

for (i in cons_vote) {
  print(paste("Judge ", cons_vote$Judge, " has a Conservative Vote Rate of: ", cons_vote$Probability))
}

# Top 6 Conservative Votes
# REMINDER! Change variable name. Top 6, not 5.
top5ConservativeProb <- head(cons_vote$Probability)
# Type check
str(top5ConservativeProb)
# It's a numeric type
boxplot(top5ConservativeProb, main = 'Five Most Conservative Judges')

# plot3 <- ggplot(top5_conservative, aes(x = Judge, y = Rate)) +
#   geom_boxplot()
plot3 <- ggplot(cons_vote, aes(x = Judge, y = Prob)) +
  geom_col()
plot3

# New data frame to store just the top 6 conservative judges
# and their voting rate.
df2 <- data.frame(head(cons_vote$Judge), head(cons_vote$Prob))
colnames(df2) <- c('Judge', 'Prob')

# Will add Gorsuch who is the judge of interest.
gorsuch <- cons_vote[11, ]
top5_conservative <- rbind(top5_conservative, gorsuch)
# Need a subset data frame with Gorsuch's name and rate only
gorsuch2 <- subset(gorsuch, select = -c(Vote, Frequency))
# Add above variable to df2
df2 <- rbind(df2, gorsuch2)

ggplot(df2, aes(x = Judge, y = Prob)) + 
  geom_point(color="blue", alpha=2.0) +
  ggtitle('5 most conservative judges based on \ntotal votes plus Gorsuch') +
  theme(plot.title = element_text(hjust = 0.5))
  #theme(plot.title = element_text(family = "Arial", face = "bold", size = (12)))

ggplot(df2, aes(x = reorder(Judge, -Rate), y = Rate)) + 
  geom_line(stat = "identity", color="darkviolet", group = 1) + # Can also use geom_step
  ggtitle('5 Most Conservative Judges based on \nTotal Votes plus Gorsuch') +
  theme(plot.title = element_text(hjust = 0.5)) +
  geom_point() +
  xlab('Judge')

#######################################################
# Will now calculate the rate which is the conservative 
# vote divided by number of times they voted 
#######################################################

# Table with each Judge's vote and count for each vote category
table1<- df1 %>% group_by(Judge) %>% count(Vote) %>% arrange(n()) 
table1

# Now want simply the count for each judge.
# Noticed error in earlier implementations
# Consider the following filter method
filter( df1, Vote == 'Conservative' )

# Will use dplyr to filter by Conservative Votes and Judges
# conservative_votes <- df1 %>%
#   filter(Vote == 'Conservative') %>%
#   count(Judge) %>%
#   arrange(desc(n))
conservative_votes <- df1 %>%
   filter(Vote == 'Conservative') %>%
   count(Judge)
 
# Recall that integer numbers in these R pipes uses standard 'n' notation
# table2 <- df1 %>%
#   group_by(Judge) %>%
#   count() %>%
#   arrange(desc(n))
table2 <- df1 %>%
   group_by(Judge) %>%
   count()
   
total_consvote <- table2 %>% pull(n)
conservative_votes <- cbind(conservative_votes, total_consvote)
colnames(conservative_votes) <- c("Judge", "n", "TotalVotes")
conservative_votes <- conservative_votes %>% arrange(desc(n))

# initiate empty column
conservative_votes["Rate"] <- NA
# For loop to fill empty column and calculate rate.
for(j in 1:length(conservative_votes$n)) {
  conservative_votes$Rate[j] <- conservative_votes$n[j]/conservative_votes$TotalVotes[j]
}

# New data frame to store just the top 6 conservative judges
# and their voting rate.

df3 <- data.frame(head(conservative_votes$Judge), head(conservative_votes$Rate))
colnames(df3) <- c('Judge', 'Rate')
gorsuch3 <- conservative_votes[11, ]
gorsuch4 <- subset(gorsuch3, select = -c(n, TotalVotes))

# t5cj <- Top 5 COnservative Judges + Gorsuch
t5cj <- rbind(df3, gorsuch4)

plot4 <- ggplot(t5cj, aes(x = reorder(Judge, -Rate), y = Rate)) + 
  geom_line(stat = "identity", color="darkorange", group = 1) + # Can also use geom_step
  ggtitle('6 Most Conservative Judge Rate with most \nTotal Votes plus Gorsuch') +
  theme(plot.title = element_text(hjust = 0.5)) +
  geom_point() +
  xlab('Judge')
plot4

# Will begin filtering for the liberal votes by judge
lib_vote <- df1 %>% group_by(df1$Judge) %>% count("Liberal") %>% arrange(desc(n))
lib_vote

# The liberal vote dataframe is the conservative vote table.
# Going to experiment other pipes.
x <-  df1 %>% filter(df1$Vote == "Liberal")
x1 <- x %>% group_by(Judge)
x2 <- x1 %>% count("Liberal")
x3 <- x2 %>% arrange(desc(n))
colnames(x3) <- c("Judge", "Vote", "Frequency")
# The above few lines can be incorporated into one direct pipe.

# Type check
str(lib_vote)
class(lib_vote)
# Table and data frame

# Rename columns similar to conservative vote.
colnames(lib_vote) <- c("Judge", "Vote", "Frequency")

hist(x3$Frequency)

# Initialize empty column to soon store probabilities.
x3["Probability"] <- NA

# Following logic use a for loop to calculate the
# liberal voting probability for each judge and stores it
# into a new column in the data frame.
for(i in 1:length(x3$Frequency)) {
  x3$Probability[i] <- x3$Frequency[i]/total_votes
}

# will now store top 7 based on total votes and store probabilities
# t7lvp <- top 7 liberal vote probability
t7lvp <- x3[1:7, c(1,4)]
t7lvp <- t7lvp %>% arrange(desc(Probability))
# Want to plot the interquartile range IQR
boxplot(t7lvp$Probability, 
        main = 'Seven Most Liberal Voting Judges',
        col = "royalblue3",
        ylab = 'probability')

ggplot(t7lvp, aes(x = reorder(Judge, -Probability), y = Probability)) + 
  geom_line(stat = "identity", color="darkseagreen4", group = 1) + 
  ggtitle('7 Most Liberal Judgese Judges based on \nTotal Votes including Gorsuch')
  theme(plot.title = element_text(hjust = 0.5)) +
  geom_point() + 
  labs(x = 'Judge')  

  
# Now want to prepare dataset for logistical regression model.
training_data <- subset(df1, select = c(Judge, Vote))
names(training_data)
dim(training_data)
summary(training_data)
pairs(training_data)
cor(training_data)
# Gives error message because the Vote variable is qualitative

# Will fit a logit model in order to predit Vote using Judge.
logit.model <- glm(Vote ~ Judge,
             data = training_data,
             family = binomial)
summary(logit.model)

# Smallest p-value associated with Judge Hartz
# Want just the coefficients for this fitted model
coef(logit.model)

# Alternatively
#summary(logit.model)$coef
summary(logit.model)$coef[,4]

# Want prob that a judge votes Liberal given values of the predictors.
logit.prob <- predict(logit.model, type = "response")

# Print first ten probabilities
logit.prob[1:10]

contrasts(training_data$Vote)
# Above line to make sure R created a dummy variable with 1 for liberal.

# Want to make prediction as to whether a judge votes Liberal or Conservative.
# Need class predictors
# Next command will create a vector of 554 Conservative votes.
logit.pred <- rep("Conservative", 554)
# Next line will transform to Liberal all elements which predicted prob exceeds 0.50,
logit.pred[logit.prob > 0.5] = "Liberal"

# Will now use the table function to get a confusion matrix
# this matrix determines how many observations were correct or incorrect.
table(logit.pred, training_data$Vote)
# Diagonal elements of confusion matrix are correct predictions
print((282+36)/554)

# Alternatively
mean(logit.pred == training_data$Vote)

# So logistic regression predicted correctly 57.4% of time
