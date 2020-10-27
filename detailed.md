## Topic-level Causality Analysis
From topic modeling results -> generate topic curve over time
topic's coverage on each time unit (e.g. one day)

Consider a weighted coverage count. Specifically compute the coverage of a topic in each document
p(topic_j|document_d) => theta_j 
Estimate the coverage of topic j at t_i
t_c_i_j as sum of thera_j over all documents with t_i timestamp.

t_c_i_j = Sum(theta_j)
TS_j => list of t_c_j for all the timestamps

creates topic stream time series that, combined with non-textual time series data
lends itself to standard time series causality measures C and testing.

Select lag value is important?
How => chose lag with highest significance


## Word-level causality and Prior Generation
Chose topics with highest causality scores and further analyze the words withing each topic
to generate topic proors

For each word, generate a word count stream W S_w by counting frequencies in the input
document collection for each day:

w_c_i = Sum c(w,d) 

Measure correlation and significance between word streams and external non-textual time series.

Then wemeasure correlations and significance between word streams andthe external non-textdual time series. This identifys words that aresignificantly correlated and their impact values.