Time series no missing gaps in the NY Times corpus

1. We will search the corpus for paragraphs where we have Bush or Gore mentioned.
2. Use only those paragraphs


May-2000 => 1May  31st May


3. There may be case where for a day, there is no news on Bush and Gore. (Ignore)
4. Day, 1196039, text (All the parargaphs where bush or Gore) => Stopwords removed, stemming done


5. LDA, tn=30

6. Output tn=30 topics(number of components), LDA object itself


Granger Test:
Pre-processing Non-text time series

“normalized” price of one candidate as a forecast probability of the election outcome: (Gore price)/(Gore price + Bush price).

Date, Price = 0.55/(.55+.5) => Gore

Date, Price => NonText 



LDA object from step-6.
Theta_j(document_topic_probability) from lda for each document.
Document_Id => Assigned by LDA when we run the algorithm. Which document comes first is assigned 0, 1, 2..


We start from May-1st:
	We find all the documents that are in May-1, from list
		# iterating over topic
		TC += Theta_j(document_id)
		Date, TC
		
New Time Series:

Compare the dates between these two
=> If we have missing data, used the future data in non-text series

02-May-2000 => 0.92  

Non-text => Keep going 1 day till we find a price => asign the price for missing date.

9. We have two time series, equal enteries


10. Take three days moving average on each series 
1, ...,n (Both the series)

X(t) - X(t-1)

1,...,n-1 enteries for the time series

11. Feed to granger test. Max_lag=5

12. Find the best lag 2

13. Significance => 1 - p-value (Time series index 0)

14. Impact_value => 


# Prior generation
14. T, Significance

15. For each topic, For each word, for each day, count for that word in documents for that day.

16. Time series, Count(W) => Index (Topic)

17. Feed to granger test (This new TS, Gore TS)

18. Word => Significance, Impact

19. Impact (50/50) => Topic has to be split =>  Tn = Tn + 1

20. Impact < 10% => Remove those words altogether => Assigning 0 probe

21. Calculate prior for all the words (doc_word_probability). eta

=> Tn , eta
=> lda start again

Iterate for 5 times

=> Table with the significant topic => words