# Iterative Topic Modeling Framework with Time Series Feedback

## Algorithm
### Parameters
#### Time series data
X = x_1, ... , x_n with timestamp (t_1, t_2, ..., t_n)

#### Collection of documents with ts from same period
D = {(d1,td1),..,(dm,tdm)}

#### Topic modeling method M
Identifies topics

#### Causality measure C
Significance measures (e.g. p-value) and impact orientation


#### tn
How many topics to model

#### mu μ
strength of the prior

#### Gamma 𝛾
Significance Threshold

#### Delta δ
Impact Threshold

### Output
k potentially causal topics
(k<=tn): (T1,L1),... (Tk, Lk)


### Steps 
![Algorithm Steps](./Algorithm.png)
1. Apply M to D to generate tn topics T1,..,TN
2. Use C to find topics with significane value sig(C,X,T) > gamma(95%)
   CT: set of candidates causal topics with lags {(tc1, L1),..,(tck,Lk)}.
3. For each candidate topic CT, apply C to find most significant
   causal words among top words w subset of T.
   Record the impact values of these significance words (e.g. word-leave Pearson 
   correlations with time series variable)
4. Define a prior on the topic model parms using significant terms and impact values
   1. Separate positive impact terms and negative impact terms
      If orientation is very weak ( delta < 10%) ignore minor group
   2. Assign prior probabilities proporations according to significance levels
   
5. Apply M to D using prior obtained in step 4 
6. Repeat 2-5 until satisfying stopping criteria (e.g. reach topic quality at some point,
no more significant topic change). When the process stops, CT is the output causal topic
list.

### Instructions for running the code:
https://github.com/sonalsharma5990/CourseProject/tree/main/src/README.md

### DataSet
https://github.com/sonalsharma5990/CourseProject/tree/main/data/README.md

 
### Major Features:
   1)Gaining access to New York Times Corpus dataset and parallel time series from Yahoo Finance.\
   2)Finding the libraries required for implementation\
   3)Perform preprocessing of data\
   4)Perform LDA on news data set  
   5)Apply Granger Test to determine causality relationship.  
   6)For each candidate topic apply Causality measure to find most significant causal words among top words in each Topic.  
   7)Record the impact values of these significance words using Pearson correlations   
   8)Separate positive impact terms and negative impact terms  
   9)If orientation of words in prior step is very weak, ignore minor group  
   10)Assign prior probabilities proportions according to significance levels  
   11)Apply LDA to Documents using prior obtained  
   12)Repeat until satisfying stopping criteria (e.g. reach topic quality at some point, no more significant topic change).  


### Hurdles and Ladders:
1) Algorithm to implement Topic Modelling:
   We had a tough call between PLSA and LDA here. 
   MP3 PLSA is heavily un-optimized. It even fails with memory error with experiment-1 document data. (Presidential campaign vs IOWA market), whereas LDA using gensim library      uses a lot of inner dependencies and the m step is not as clear(as in lectures) to incorporate Mu.(question @1378 on Piazza)
   Post feedback and discussion with Professor and Students we used the decay parameter as Mu to implement the paper.

2) Missing data for some dates in Non-text series 
   We have used future value in this case after research.To justify the same, in case of stock data in week 9/11, we would miss the impact in stock if using previous values.
 
3) Granger Test to determine causality relationship
   We used 1- p value for score which amounts to almost 0 values getting 100% score.

4) Add customized stop words in data preprocessing
   We removed words that were not adding any value to topics found.
      * names of candidates as they are frequently used
      * political jargon words
      * parties
      * states
      * common verbs and words
      * time words


#### Final Results:

