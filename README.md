# Iterative Topic Modeling Framework with Time Series Feedback

Please fork this repository and paste the github link of your fork on Microsoft CMT. Detailed instructions are on Coursera under Week 1: Course Project Overview/Week 9 Activities.

## Algorithm
### Parameters
#### Time series data
X = x_1, ... , x_n with timestamp (t_1, t_2, ..., t_n)

#### Collection of documents with ts from same period
D = {(d1,td1),..,(dm,tdm)}

#### Topic modeling method M
Identifies topics

#### Casuality measure C
Significance measures (e.g. p-value) and impact orientation


#### tn
How many topics to model

#### mu μ
strength of the prior

### Output
k potentially casual topics
(k<=tn): (T1,L1),... (Tk, Lk)


### Steps
![Algorithm Steps](./Algorithm.png)
1. Apply M to D to generate tn topics T1,..,TN
2. Use C to find topics with significane value sig(C,X,T) > gamma(95%)
   CT: set of candidates casual topics with lags {(tc1, L1),..,(tck,Lk)}.
3. For each candidate topic CT, apply C to find most significant
   casual words among top words w subset of T. 
   Record the impact values of these significance words (e.g. word-leave Pearson 
   correlations with time series variable)
4. Define a prior on the topic model parms using significant terms and impact values
   1. Separate positive impact terms and negative impact terms
      If orientation is very weak ( delta < 10%) ignore minor group
   2. Assign prior probabilities proporations according to significance levels
   
5. Apply M to D using prior obtained in step 4 
6. Repeat 2-5 until satisfying stopping criteria (e.g. reach topic quality at some point,
no more significant topic change). When the process stops, CT is the output casual topic
list.

## Tasks
1. Find suitable topic modeling method with prior (Use EM?) :Sonal
2. Find suitable Casuality Measure : Kamleesh
3. Find how to calc gamma
4. How to identify +, - impact terms (Use pearson coefficients)
5. Find how to calc delta
6. Calculate Prior : Maneesh
7. Identify suitable lag
