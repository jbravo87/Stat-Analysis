####################
# First part will focus on exploring some simple questions.
# 1. With what frequency did each respective judge vote conservatively
#    on immigration cases in the first vote (Vote1)?
# Will implemeny descriptive statistic
####################

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
ggplot(cons_vote, aes(x = cons_vote$Frequency)) + geom_bar(stat = "identity")

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

# The following lines uses a for loop to calculate the
# conservative voting rate for each judge and stores it
# into a new column in the data frame.
for(i in 1:length(cons_vote$Frequency)) {
  cons_vote$Rate[i] <- cons_vote$Frequency[i]/total_votes
}

for (i in cons_vote) {
  print(paste("Judge ", cons_vote$Judge, " has a Conservative Vote Rate of: ", cons_vote$Rate))
}

# Top 5 Conservative Votes
top5ConservativeRate <- head(cons_vote$Rate)
# Type check
str(top5ConservativeRate)
# It's a numeric type
boxplot(top5ConservativeRate, main = 'Five Most Conservative Judges')

# plot3 <- ggplot(top5_conservative, aes(x = Judge, y = Rate)) +
#   geom_boxplot()
plot3 <- ggplot(top5_conservative, aes(x = Judge, y = Rate)) +
  geom_col()
plot3

# New data frame to store just the top 6 conservative judges
# and their voting rate.
df2 <- data.frame(head(cons_vote$Judge), head(cons_vote$Rate))
colnames(df2) <- c('Judge', 'Rate')

# Will add Gorsuch who is the judge of interest.
gorsuch <- cons_vote[11, ]
top5_conservative <- rbind(top5_conservative, gorsuch)
# Need a subset data frame with Gorsuch's name and rate only
gorsuch2 <- subset(gorsuch, select = -c(Vote, Frequency))
# Add above variable to df2
df2 <- rbind(df2, gorsuch2)

ggplot(df2, aes(x = Judge, y = Rate)) + 
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
