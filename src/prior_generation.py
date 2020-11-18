"""Word level causality and prior generation."""


## filter topics with highest causality score


# for each significant topic, for each word, generate word_count stream
# by counting frequencies in the input document collection for each day.

# measure correlation and significance between word streams
# end external non textual time series

# generate topic priors for significant words in significant topics
# topic prior => dirichlet distribution
# assigns high probabilities to significant words.

# consistency: chose topic that have mostly positive or negative corelation
# if one topic has both '+' and '-' corelation => separate into two topics
# i number_of_positive < 0.1* number of negative, keep probability of another
# group 0.

## Cumulative probability mass cutoff


## input would be topics with causality score