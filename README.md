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

## Tasks
1. Find suitable topic modeling method with prior (Use EM?): Sonal (Would take the XML and try generating topics)
2. Find suitable Causality Measure : Kamlesh (Continue coding and research)
3. Find how to calc gamma: Sonal 
4. How to identify +, - impact terms (Use pearson coefficients): Maneesh (Continue coding/research)
5. Find how to calc delta: Maneesh
6. Calculate Prior : Maneesh
7. Identify suitable lag

## Dataset Tasks
~~1. We examine the 2000 U.S. Presidential elec-
tion campaign. The input text data comes from New York Times
Iowa Electronic Markets (IEM) 3 2000 Presidential Winner-Takes-
All Market
How to define relation?
Dates: 11/01/2000 11/10/2000~~

~~2. stock prices of American Airlines
and Apple and the same New York Times text data set with longer
time period, but without keyword filtering, to examine the influence
of having different time series variables for supervision.~~

~~Dates: NY Times July 2000 through December 2001 as the text input.
Search for Stock prices till July 2000 to December 2001
Research Part: Kamlesh, Sonal, Maneesh~~

## Final Deliverables
1. Your documented source code and main results. (Piazza)
2. A demo that shows your code can actually run on the test dataset and generate the desired results. You don’t need to run the training process during the demo. If your code takes too long to run, try to optimize it, or write some intermediate results (e.g. inverted index, trained model parameters, etc.) to disk beforehand.
3. Discuss how your results match or mismatch those reported in the original paper. Your results should cover all the main aspects and datasets discussed in the paper.
4. If some of your results do not match the paper, discuss possible reasons and solutions.
