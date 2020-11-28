This folder is to try different things

## main.py
- Load Doc, Date mapping file as vector of length D
    - ith entry in the vector is document i date
    - D is number of documents
    - variable name in code: doc_date_map

- Load IEM data file. Standardize price of one candidate (Gore)
    and take only Gore entries.
    - It's a pandas dataframe with two columns.
    - Date, LastPrice
    - variable name in code: iem_data


## Topic Causality Analysis flow
- Extract document topic probability (theta) from LDA model as matrix.
    - theta =  dimension D * T
    - D count of all documents, T count of all topics
    - i, j entry is probability of document i belonging to topic j.
    - variable name in code: topics.

- Use Date column for IEM data. It is a vector of size n.
    - i the entry is date for sample i.
  

- Combine vector IEM dates (length n) and doc date mapping (length D)
    to create adjancy matrix of size D * n
    - i, j is 1 only if document i exists for non-text date j
    - variable name in code: common_dates

- Take Transpose of theta and matrix multiply with common_dates
    - Theta.T @ common_dates
    - Dimensions wise
        ```
        = Theta.T @ common_dates
        = (T * D) @ (D * n)
        = (T * n)
        ``` 
    - In the result all theta for documents are summed for each day
    - We get a timeseries from Topic i in ith row. Every jth entry is
        Theta_sum for each day
    - The timeseries length n is same as IEM data length
    - variable name in code: time_series

- Pass each row (Topic wize) of time_series to granger test along with IEM price column