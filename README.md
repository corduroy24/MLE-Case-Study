# Priceline Machine Learning Case Study

## Summary of Approach & Findings  


This study aims to predict which hotels to display based on which has the highest likelihood of being booked by a user. 




**Approach**
- Data Understanding and EDA:
    - `destinationName` had 2.43% missing values. After exploring how to best handle the missing data, I made the assumption that that a user either had `destinationName` filled for all interactions or they didnt. So to handle this, it was decided to fill the missing values with "Unknown". 
    - Significant class imbalance exists in the dataset (ie., not booked = 740,497, booked = 28,612`). In other words, approx 4% booked. To remedy this class imbalance,  Upsampling via SMOTE was used. 
- Preprocessing and Feature Engineering
    - Dates were removed since the format cannot be used. Features were extracted from the dates, such as `days_until_checkin` and `stay_duration`
    - `clickLabel` was removed because it was highly correlatd with `bookingLabel`. In other words, both variables shared the same information. This can be validated since I first trained a model with clickLabel involved and it was the only important feature detected  
    - Data processing was done in PySpark due to data import and feature exploration, while model training was done using Scikit-Learn & Pandas flexibility and SMOTE. Note that, since the dataset isnt over 2 Million rows, pandas is still effective. Also, XGBoost enables use all cores.   
    - One hot encoding was used for `vipTier` and `deviceCode` because these variables did not have many categories
    - label encoding was utilized for `destinationName` because if one hot encoding was used, it would create a sparsed dataset, that would be longer to train. Also note that XGBoost does not treat numbered categories as ordinal values. 
    - brandId, hotelId, searchId, and userId were dropped because these columns were found to be redundant (low predictive value). Notice that the number of unique categories for each feature exceeded 10000. So it is assumed that the effects would not be signficant. 
- Model Development
    - Target variable (`bookingLabel`) indicates that this is a classification problem 
    - XGBoost was mainly considered over Logistic regression because it can handle high-cardinatality like `destinationName`. Note that, logistic regression would allow for effective interpreation of the model and data insights. The model also trains faster than RandomForest. 
    - Hyperparameter tuning was done via Randomized Search for speed 
    - Due to time constraints, only one model was trained. Ideally, 2 or more models would have been compared. 
- Model Evaluation
    - AUC-ROC (main): helps to understand the models ability to seperate bookings from non-bookings (AUC Score: 0.9938)
    - Precision: is an effective metric to look at because it signals how well the model provides relevant hotels given the users intent
    - Recall: helps to prevent missed bookings
    - Top 10 important features: `rank`, `reviewCount`, `destinationName`, `customerReviewScore`, and `minPrice`. This validates that the order of the hotel listings has a significant impact on whether or not a person would book. So priceline already influences what hotels users book. 




## Recommendations for improving the system
- Using PySpark ML for preprocessing and model training would enhance speed and scalability in a production environemt.
- instead of auto sampling strategy, use of 30% ratio sampling to mitigate overfitting on sythetic data
- Structured logs for monitoring and debugging as opposed to print statements
- Optimized hyperparamter tuning, by considering more parametrs, gridsearch or Bayesian optimization 
- model versioning / tracking 
- set up experiments for performance tracking
- add hotel specific features (ie., total booking per hotel)
- remove minStrikePrice, since is it highly correlated with minPrice
- fill `destinationName` by most popular destination per user otherwise use "Unknown"
- Docstrings for functions
- environment configurations
- checks for other missing values in the pipeline and other data validation
- make use of try-except blocks to catch errors during execution.
- optimize threshold 
- use SHAP values for feature importance and better interpretation
- deploy model usng google vertex AI for real time inference
- add code comments

## Instructions for Setup & Execution  
- pip install -r requirements.txt
- python version 3.10.9
- To run the pipeline: python pipeline.py
- Check the model metrics (ie., model_metrics.json)



