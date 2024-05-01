# Optimizing EV Charging Station Performance through Predictive Analytics and Simulation
Project to Showcase interest in stable auto and personal skills


## Project Objective:
Develop a comprehensive analysis and simulation model to predict and improve EV charging station performance, optimize operations, and enhance revenue generation strategies.

## Built-with: 
* Python[3.12.3] 
    * Pandas[2.2.2]
    * Numpy [1.26.4]
    * Scikit-learn 
* SQL [PostgreSQL]
* Docker

## Project Outline: 
I. Data Preparation and Exploration 
  * Data Cleaning: Process dataset to handle missing values, correct anomalies, and prepare it for analysis.
  * Exploratory Data Analysis: Use visualizations to understand the data’s underlying patterns, distributions, and correlations.
    
II. Predictive Modeling
  * Demand Forecasting Model: Build a machine learning model to predict daily usage patterns and demand at EV charging stations. Techniques include time series analysis(linear and ARIMA) and regression models.
  * Performance Prediction Model: Develop a model to predict the energy consumption of charging stations.
    
III. Simulation of Charging Station Operations
  * Capacity Planning: Use simulations to model scenarios where demand may exceed supply, and identify the impact of adding more charging stations.
  * Pricing Strategy Simulation: Simulate different pricing models to find the optimal balance between usage and revenue.
    
IV. Data Pipeline and Monitring 
  * Data Pipeline Design: Sketch out a data pipeline that can automate data collection, processing, and feeding into machine learning models for real-time performance monitoring.
  * Performance Dashboard: Develop an interactive dashboard that displays real-time analytics and KPIs to monitor charging station performance.





## Dataset 
This [dataset](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/NFPQLW) contains information from 3,395 high resolution electric vehicle charging sessions. The data contains sessions from 85 EV drivers with repeat usage at 105 stations across 25 sites at a workplace charging program. The workplace locations include facilities such as research and innovation centers, manufacturing, testing facilities and office headquarters for a firm participating in the U.S. Department of Energy (DOE) workplace charging challenge. The data is in a human and machine readable *.CSV format. The resolution of the data is to the nearest second, which is the same resolution as used in the analysis of the paper. It is directly importable into free software. (2020-07-30)

| Columns        | Description                                                                                                         |
|----------------|---------------------------------------------------------------------------------------------------------------------|
| sessionId      | Charging session ID                                                                                                 |
| kwhTotal       | Total energy use (kWh)                                                                                              |
| dollars        | Amount paid by user (USD)                                                                                           |
| created        | Charge start date and time (00YY-MM-DD HH:MM:SS)                                                                    |
| ended          | Charge end date and time (00YY-MM-DD HH:MM:SS)                                                                      |
| startTime      | Charge start hour (military time)                                                                                   |
| endTime        | Charge end hour (military time)                                                                                     |
| chargeTimeHrs  | Total charge time (hr)                                                                                              |
| weekday        | Day of week                                                                                                         |
| platform       | Platform used to log session information by EV user                                                                 |
| distance       | Distance from a user's home to the charging location, expressed in miles except where user did not report address   |
| userId         | User ID                                                                                                             |
| stationId      | Station ID                                                                                                          |
| locationId     | Location ID                                                                                                         |
| managerVehicle | Firm manager vehicle indicator                                                                                      |
| facilityType   | Type of facility a station is installed at (manufacturing = 1, office = 2, research and development = 3, other = 4) |
| Mon            | Monday indicator                                                                                                    |
| Tues           | Tuesday indicator                                                                                                   |
| Wed            | Wednesday indicator                                                                                                 |
| Thurs          | Thursday indicator                                                                                                  |
| Fri            | Friday indicator                                                                                                    |
| Sat            | Saturday indicator                                                                                                  |
| Sun            | Sunday indicator                                                                                                    |
| reportedZip    | Zip provided by user indicator                                                                                      |

## Dataset Preparation and Exploration 
1. Converting data and time fields to python datetime 
    * 'created' and 'ended' 
    * The year column in the time is wrong i.e. "0014-11-18 15:40:26" should actually be 2014 rather than 0014 (Also for 0015 as well)

2. Check for missing values
    * 'distance' is the distance from a user's home to the charging location, expressed in miles except where user did not report address
    * We see that the only column that has missing/null values is the column 'distance' which is expected as this is a user-reported data. We have a few ways to go at this:
      1. leave as-is
      2. fill with placeholder 
      3. impute values 
          * mean/median imputation
          * model based imputation 
      4. remove entries

For now I will choose to leave as-is because it's not critical for my primary analysis/simluations. This avoids introducing potential biases. 

3. Feature Engineering
     * We do see that a lot of features such as day of the week and hour are available so I will leave this as is for now.

## EDA 

Histogram for distribution of key metrics for 'kwhTotal' and 'chargeTimeHrs'.
<img src="imgs/histogram.png" width="800"/>

Time series graph for 'kwhTotal' trends over time.
<img src="imgs/timeSeries.png" width="800"/>

Distribution of 'kwhTotal' across days of the week.
<img src="imgs/days.png" width="800"/>

Correlation heatmap to examine correlation between numeric features.
<img src="imgs/corrMatrix.png" width="800"/>


## Predictive Modeling

1. Performance Prediction Model

<img src="imgs/performancePred.png" width="800"/>

* MAE = 1.66
* MSE = 10.00
* R² = -0.31

Analysis:
* MAE and MSE: The MAE and MSE values appear relatively low, but this assessment really depends on the scale of your target variable (kWh). If the typical values of your target are large, these errors might be acceptable.
* R²: The negative R² indicates that the model performs worse than a horizontal mean line. This is a sign that the model is not capturing the underlying pattern and might be poorly specified or using non-relevant features.

Improvements:
* Feature Engineering: Investigate adding more relevant features or transforming existing features to capture non-linear relationships.
* Model Complexity: Consider using more complex models that can capture more complex patterns in the data, such as random forests or gradient boosting machines.
* Data Quality: Ensure the data quality is high and check for any data preprocessing issues like outliers or incorrect data entries that could be influencing model performance.


2. Demand Forecasting with Simple Linear Regression

<img src="imgs/demandForeLinear.png" width="800"/>

* MAE = 52.00
* MSE = 4170.05
* R² = 0.17

Analysis:
* MAE and MSE: These values are high, indicating that on average, the model’s predictions are off by 52 kWh, with some errors squared, leading to a higher MSE. Given the typical scale of daily kWh demand, these errors might be significant.
* R²: The value of 0.17 suggests that only 17% of the variance in the kWh total is being explained by the model, which is quite low. This indicates a poor fit.

Improvements:
* Model Type: Linear regression may be too simplistic to capture the dynamics in the kWh data, especially if there are non-linear relationships.
* Additional Features: Include time-based features like day of the week, holidays, or weather conditions, which can impact energy usage.
* Higher-Order Terms: Try polynomial regression or interaction terms to capture more complex relationships.

3. Demand Forecasting with ARIMA

<img src="imgs/demandForeARIMA.png" width="800"/>

* MAE = 73.12
* MSE = 7324.85
* RMSE = 85.59

Analysis:
* MAE and RMSE: The MAE and RMSE are quite high, which indicates significant forecasting errors. Given that these values are likely to represent daily totals, errors of this magnitude can be quite impactful.
* Model Fit: The high errors suggest that ARIMA may not be capturing all the relevant patterns, particularly if there is seasonality or non-stationary behavior in the dataset that hasn't been addressed adequately.

Improvements:
* Check Stationarity: Revisit the stationarity assumption and make sure the data are properly differenced to achieve stationarity.
* Seasonal Model: If there is seasonality, consider using a SARIMA model which incorporates both non-seasonal and seasonal elements.
* Model Diagnostics: Review the residuals of the ARIMA model for any patterns that suggest poor model fit and adjust the model parameters accordingly.

