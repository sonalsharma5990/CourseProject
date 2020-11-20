# Instructions for running the code

## Modules
Run below for
- mu = 10,50,100,500,1000
- Tn = 10,20,30,40, TnVar(The point where causal topics starts to fall relative to previous iteration)
- Lag Value

### LDA
- Input: 
    - Document D1,..DT
    - Prior 
    - Collection 
- Config: Tn: Number of topics
- Output: Tn Topics containing words

### Casualty Score
- Input: 
    - Topics containing words
    - Lag Value
- Output: 
    - Topics containing words with Causality score(significance score)
    - If lag is not 1 day, cross validate the topics for all lags 1 to lag
    and choose the lag with highest significance
### prior_generation
- Input: Topics with causality significance score
- Configuration:
    - mu (Strength of prior)
    - probM 
- Output: 
    - Topic prior
    - Topics containing words with impact and significance

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
