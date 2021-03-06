# Instructions for running the code

## Modules
Run below for
- mu = 10,50,100,500,1000
- Tn = 10,20,30,40, TnVar(The point where causal topics starts to fall relative to previous iteration)
- Lag Value

### pre_process
This module does the filtering and creating corpus.
pre_process and create non-text time series. Fill in missing dates
with average/future values.

### LDA
- Input: 
    - Document D1,..DT
    - Prior 
- Config: Tn: Number of topics
- Output: trained lda model

### Casualty Score
This module 

- Input: 
    - LDA model
    - Significance cutoff gamma
- Output: 
    - Topics with lags
        Tuple containing topic_id, lag

### prior_generation
- Input: 
    - Topics with lags
    - LDA model
    - corpus
- Configuration:
    - mu (Strength of prior)
    - probM
    - Significance cutoff gamma
- Output: 
    - topic word prior

### measure of quality
Calculate purity


### out_plot
- Input: List of average causilty confidence and purity for each mu and tn
- Generates output plots for mu and t_n


## Install dependencies
```
pip install -r requirements.txt
```

## Run main.py
```
python main.py
```

## Run Tests
```
python run_tests.py
```
