[![LinkedIn][linkedin-shield]][linkedin-url]

# Optimizing EV Charging Station Performance through Predictive Analytics and Simulation

<p align="center">
   <img src="imgs/960x0.webp" width="360"/>
</p>


## Project Objective:
Develop a comprehensive analysis and simulation model to predict and improve EV charging station performance, optimize operations, and enhance revenue generation strategies.

## Built-with: 
* Python[3.12.3] 
    * Pandas[2.2.2]
    * Numpy [1.26.4]
    * Scikit-learn [1.4.2]
    * Simpy [4.1.1]

## Project Outline: 
I. Dataset Preparation and Exploration
  * Data Cleaning: Process dataset to handle missing values, correct anomalies, and prepare it for analysis.
  * Exploratory Data Analysis: Use visualizations to understand the data’s underlying patterns, distributions, and correlations.
    
II. Predictive Modeling
  * Demand Forecasting Model: Build a machine learning model to predict daily usage patterns and demand at EV charging stations. Techniques include time series analysis and regression models.
  * Performance Prediction Model: Develop a model to predict the energy consumption of charging stations.
    
III. Simulation of Charging Station Operations
  * Capacity Planning: Use simulations to model scenarios where demand may exceed supply, and identify the impact of adding more charging stations.
  * Pricing Strategy Simulation: Simulate different pricing models to find the optimal balance between usage and revenue.
    
IV. Data Pipeline and Monitering 
  * Data Pipeline Design



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


***Quick Note*** 
This dataset, spanning 2014 to 2015, captures a period when electric vehicles (EVs) were less common than they are today but the sales were increasing significantly. Historically, EV sales have risen exponentially from 2011 onwards, reflecting a significant shift in consumer adoption and technological advancement in the automotive industry. The graph of vehicle sales from 2000 to 2021, sourced from the [Bureau of Transportation](https://www.bts.gov/), underscores this trend of rapid growth. 

<img src="imgs/Screen Shot 2024-05-01 at 10.58.11 AM.png" width="800"/>

The exponential nature of this growth presents challenges for predictive modeling, particularly when using historical data from periods of relatively lower market penetration and sales volume. This exponential growth leads to data volatility, where earlier data may not accurately represent more recent trends, thereby affecting the performance of models trained on this dataset. Models may struggle to forecast future trends accurately if they cannot adjust to the accelerating pace of sales and the evolving market dynamics. Such discrepancies can introduce flaws in predictions, as the model may not fully account for the rapidly changing market conditions and consumer behaviors driving the surge in EV popularity.


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

### Histogram for distribution of key metrics for 'kwhTotal' and 'chargeTimeHrs'
<img src="imgs/histogram.png" width="800"/>

#### Key Observations(kWh)
1. Shape
   * The distribution is heavily skewed to the right, with a peak around 7-8 kWh and a long tail extending to higher kWh values. This suggests that most charging sessions involve small amounts of energy, with fewer sessions consuming more energy.
2. Central Tendency
   * The mode of the distribution (the most frequent value) is around 7-8 kWh, indicating that this is a common amount of energy drawn during a session.
3. Variability
   * There is significant variability in the distribution, as indicated by the long tail. Some charging sessions consume substantially more energy than the typical amounts.

#### Key Observations (Charge Time)
1. Shape
   * The distribution of charging time is also right-skewed, with a peak around 4-5 hours and a tail extending to longer durations.
2. Central Tendency
   * Most charge times are short, with the mode between 4-5 hours.
3. Variability
   * There is a presence of outliers(12.4, 15.2, 55.2 hours!) that last significantly longer than the typical 1-2 hours, though these are less frequent. These may be worth looking into as they charged longer than 10 hours. 


### Time series graph for 'kwhTotal' trends over time.
<img src="imgs/timeSeries.png" width="800"/>

#### Key Observations
1. Seasonal Trends:
   * There appears to be a cyclic pattern in the data, with fluctuations that might suggest seasonal trends. This could indicate higher usage in certain months, possibly influenced by weather conditions, holiday seasons, or other seasonal factors.
2. Volatility:
   * The data shows significant volatility in daily kWh usage, with some days experiencing very high usage and others much lower. 
3. Growth Over Time:
   * There seems to be an upward trend in the latter part of the series, particularly noticeable starting around early-2015. This could be indicative of an increase in the adoption of EVs, expansion of charging infrastructure, or both.
4. Dips and Peaks:
   * Specific notable dips and peaks could be linked to external events. For instance, the sharp dips around early 2015 might warrant further investigation to understand underlying causes.

### Distribution of 'kwhTotal' across days of the week.
<img src="imgs/days.png" width="800"/>

#### Key Observations
1. Variability and Median Values:
   * Monday through Friday show similar median kWh usage values, with Tuesday having a slightly lower variability in kWh usage as indicated by the shorter interquartile ranges. This suggests a more consistent usage pattern throughout the weekdays.
   * Saturday and Sunday show significantly higher variability in kWh usage with higher median values, particularly on Sunday. Sunday shows not only the highest median kWh usage but also the broadest range of kWh usage, as indicated by the longer box and more spread out outliers.
2. Outliers:
   * There are numerous outliers on most days, but except on Sunday. Outliers indicate unusually high or low kWh usage on these days.
   * The presence of outliers, especially on Saturday, suggests that there can be extreme variations in usage, possibly due to non-routine activities or events.
3. Spread and Dispersion:
   * The weekend (Saturday and Sunday) boxes are not only taller (indicating a higher IQR), but also show the tails extending further from the median than on weekdays. This could imply less predictability in usage patterns during these days.

### Correlation heatmap to examine correlation between numeric features.
<img src="imgs/corrMatrix.png" width="800"/>

#### Key Observations
1. kWhTotal and dollars (0.53):
   * This moderate positive correlation suggests that as the total kWh delievered increases, the total amount of money also tends to increase, which is expected since more energy being delivered should mean that it costs more.
2. facilityType and managerVehicle(-0.20):
   * A correlation of -0.20 suggests a weak inverse relationship, possibly indicating different types of facilities are used by different types of vehicles or management styles.
3. facilityType and several days of the week (ranging from -0.19 to 0):
   * These correlations might suggest varying usage patterns on different days of the week depending on the facility type. For instance, certain facilities might see more use during weekdays if they are near workplaces.


## Predictive Modeling

1. Performance Prediction Model - Simple Linear Regression

<img src="imgs/performancePred.png" width="800"/>

* MAE = 1.66
* MSE = 10.00
* R² = 0.31

Analysis:
* MAE and MSE: The MAE and MSE values appear relatively low, but this assessment really depends on the scale of your target variable (kWh). If the typical values of your target are large, these errors might be acceptable.
* R²:
   * The R² score of 0.31 means that the model explains 31% of the variance in the dependent variable based on its input variables. This is a measure of how well the differences in the independent variables account for the variations in the dependent variable you're trying to predict.
   * This indicates a low to moderate level of fit. While the model does provide some insight into the data, a significant portion of the variance (69%) remains unexplained by the model. This suggests that there may be other factors or variables not included in the model that influence the dependent variable.

Improvements:
* Feature Engineering: Investigate adding more relevant features or transforming existing features to capture non-linear relationships.
* Model Complexity: Consider using more complex models that can capture more complex patterns in the data, such as random forests or gradient boosting machines.
* Data size: Consider using more data from a more recent time where EV use have spiked.

2. Demand Forecasting - Simple Linear Regression

<img src="imgs/demandForeLinear.png" width="800"/>

* MAE = 52.00
* MSE = 4170.05
* R² = 0.17

Analysis:
* MAE and MSE: These values are high, indicating that on average, the model’s predictions are off by 52 kWh, with some errors squared, leading to a higher MSE. Given the typical scale of daily kWh demand, these errors might be significant.
* R²: The value of 0.17 suggests that only 17% of the variance in the kWh total is being explained by the model, which is quite low. This indicates a poor fit.

Improvements:
* Model Type: Linear regression may be too simplistic to capture the dynamics in the kWh data, especially if there are non-linear relationships.
* Higher-Order Terms: Try polynomial regression or interaction terms to capture more complex relationships.

3. Demand Forecasting - ARIMA

<img src="imgs/demandForeARIMA.png" width="800"/>

* MAE = 73.12
* MSE = 7324.85
* RMSE = 85.59

Analysis:
* MAE and RMSE: The MAE and RMSE are quite high, which indicates significant forecasting errors. Given that these values are likely to represent daily totals, errors of this magnitude is quite impactful.
* Model Fit: The high errors suggest that ARIMA may not be capturing all the relevant patterns, since there is seasonality and non-stationary.
  
Improvements:
* Seasonal Model: Since there is seasonality, consider using a SARIMA model which incorporates both non-seasonal and seasonal elements.
* Model Diagnostics: Review the residuals of the ARIMA model for any patterns that suggest poor model fit and adjust the model parameters accordingly.


## Simulation of Charging Station Operations ## 
*Didn't have much time to complete this part fully but wrote up what I have in mind*

We can simulate using the historical data we have from our original dataframe. Here are some important numbers to keep in mind: 
* Average number of sessions per hour: 2.2088484059856865
* Average charge duration: 170.48925871379478 minutes
* Standard deviation of charge duration: 90.4483147118524 minutes

We can then use these metrics create a simple simulation using packages in Python such as [Simpy](https://simpy.readthedocs.io/en/latest/). Simpy is great for simulating variables with discrete distributions.

For the simulation we can get these outputs using the methods listed: 
1. Arrival Times - Exponential Distribution
   * Given that vehicle arrivals at charging stations can be reasonably assumed to be independent events with a constant mean rate, using the exponential distribution is justified.
2. Charge Duration: Normal Distribution
   * Central Limit Theorem: This theorem suggests that when independent random variables are added, their properly normalized sum tends toward a normal distribution even if the original variables themselves are not normally distributed. In most situations, the total charge time can be affected by small independent factors; battery level, charger type, etc... however, these can all be approximated by a normal distribution.

We can build something simple like this at first to take an initial stab at simulating with our parameters.

```
import simpy
import random

def ev_charging_station(env, number_of_chargers, arrival_rate):
   """Simulation of charge station"""
    charger = simpy.Resource(env, number_of_chargers)
    
    while True:
        yield env.timeout(random.expovariate(arrival_rate)) # Exponential distribution with Arrival times
        env.process(vehicle(env, charger))

def vehicle(env, charger, mean_duration, std_duration):
    """Simulate the charging process of a single vehicle"""
    with charger.request() as request:
        yield request
        charge_duration = random.normalvariate(mean_duration, std_duration)  # Normal distribution with Charge duration
        yield env.timeout(charge_duration)

env = simpy.Environment()
env.process(ev_charging_station(env, number_of_chargers=2, arrival_rate=1/arrival_rate_per_hour))
env.run(until=1440)  # Simulate for one day (1440 minutes)

```
## Data Pipeline and Monitering 
Assuming from the job description, the application will be utilizing RESTAPI's to grab data. I have built a sketch of a possible solution for utilizing AWS services to create a robust and scalable solution. This architecture will be able to handle the data flow from the source to storage and analysis. 

Components:
* API Gateway: Use AWS API Gateway to manage, secure, and route API calls.
* AWS Lambda: Process API requests by executing the business logic and interacting with other AWS services for data handling.
* Amazon S3: Use as the primary data lake storage for raw data collected from the API.
* AWS Glue: Perform ETL operations on the data stored in S3 to transform it into a structured format suitable for analysis.
* Amazon RDS or Amazon Redshift: Serve as the data warehousing solution to store and manage transformed data.
* Amazon QuickSight: Provide business intelligence capabilities by allowing visualization and reporting on the processed data.
* Amazon CloudWatch: Monitor the performance of all components, especially API Gateway and Lambda, to ensure operational health and log data for audits.


<img src="imgs/stable_auto_drawio.drawio.png" width="1200"/>

### Detailed AWS Architecture Flow:
1. **Data Ingestion**:
   * Client applications send data to the REST API hosted on AWS API Gateway.
   * API Gateway receives API requests and forwards them to AWS Lambda for processing.
2. **Data Processing and Initial Storage**:
   * AWS Lambda processes the incoming data (validation, transformation, etc.) and stores the raw data in Amazon S3. This storage acts as a data lake, where data is kept in its original form.
3. **ETL Processing**:
   * AWS Glue is triggered on a schedule or event (such as new data upload completion in S3). It performs ETL tasks to transform raw data into a structured format. AWS Glue can read data from S3, transform it, and then load the processed data either back into S3 in a different format or directly into a database.
   * For complex transformations, AWS Glue can use PySpark or Scala scripts, which are scalable and handle large datasets efficiently.
4. **Data Storage for Analysis**:
   * The transformed data is loaded into Amazon RDS for transactional queries or Amazon Redshift for analytics and warehousing. This choice depends on the nature of the data and the type of queries that will be performed.
5. **Data Visualization and Reporting**:
   * Amazon QuickSight accesses the data in RDS or Redshift to create visualizations, dashboards, and reports, providing insights and analytics to business users.
6. **Monitoring and Logging**:
   * Amazon CloudWatch monitors the performance of API Gateway, Lambda functions, and other services. It collects logs, metrics, and events, providing a comprehensive view of the AWS environment's health and activity.


## Contact 

Justin (Jin Wook) Lee  - justinjwlee1114@gmail.com

[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&color=blue
[linkedin-url]: https://www.linkedin.com/in/justinjwlee1114/
