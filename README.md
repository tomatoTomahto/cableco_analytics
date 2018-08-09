# Cableco Analytics
## Project Overview
This project analyzes customer information and service usage data to predict
1. Expected Customer Monthly Spend
2. Customer Churn Likelihood

## Project Setup
### Generating Customer Data
Run ```GenerateCustomerData.py``` to generate fake data for approximately 1M customers
and write it to Kudu. 

### Creating Model Data
Run ```GenerateModelData.py``` to read the customer data, create features for each
customer and write it to Kudu

## Data Analytics
### Data Exploration
Run ```BuildMLModels.py``` in a workbench with at least 2 cores, 4GB memory

### Model Development
The script ```BuildMLModels.py``` will generate 2 models:
* Linear Regression Model that predicts the monthly spend of a given customer
* Random Forest Classification Model that predicts the likelihood a given customer will churn