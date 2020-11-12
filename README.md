# Flant Spot Instance Prediction Flask API

Flask API which handles training models with continuously updated
spot pricing history and returning predictions for a given spot instance.
The date range of predictions is from the last data point provided to 30 days ahead.

## Setup
1. `$ python -m venv venv`
2. 
    - Windows: `$ ./venv/Scripts/activate`
    - Linux/Mac OS: `$ source venv/bin/activate`
3. `$ pip install -r requirements.txt`