# CIVL4910-FYP

### Python 3.13.11

### Environment:
I used Anaconda Navigator to set up the environment.

I am not sure which environments are already installed on my computer, so I listed those I believe are necessary.

1. httpx
2. neo4j
3. numpy
4. pandas
5. perplexityai
6. matplotlib
7. scikit-learn
8. transformers

### API:
I use the Perplexity API, but it just support API of Sonar model instead of Claude 4.5.

Register Perplexity using UST email and upgarde it to education version to access the API. 

Both "aaa.py" and "environment_test.env" are test files, download them and try running "aaa.py" your local device.

### Dataset
The dataset is from: https://github.com/Puw242/SafeTraffic/tree/main/data/WA/test
"inj.csv", "sev.csv" and "type.csv" record same accident cases. The only different is the role and function of "Assistant:<...>",  in each case.
"inj.csv" predicts number of people injured in the crash event, "sev.csv" predicts severity level of the crash, "type.csv" classifies type of crash.
As our project doesn't require making predictions, so I clean content related to predictions in data_loader.py.
