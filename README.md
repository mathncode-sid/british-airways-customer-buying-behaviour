# Customer Booking Prediction

## Overview
This project aims to predict whether a customer will complete a booking based on trip details, preferences, and booking patterns. The dataset contains 50,000 records with a mix of numerical and categorical features. The target variable is `booking_complete` (1 = booking completed, 0 = not completed).

## Dataset
The dataset (`customer_booking.csv`) includes the following features:
- `num_passengers`: Number of passengers
- `sales_channel`: How the booking was made
- `trip_type`: RoundTrip, OneWay, or CircleTrip
- `purchase_lead`: Number of days between booking and departure
- `length_of_stay`: Number of days between arrival and return
- `flight_hour`: Hour of departure
- `flight_day`: Day of the week for departure
- `route`: Origin and destination airport codes
- `booking_origin`: Country where the booking was made
- `wants_extra_baggage`: 0 or 1
- `wants_preferred_seat`: 0 or 1
- `wants_in_flight_meals`: 0 or 1
- `flight_duration`: Flight duration in hours
- `booking_complete`: Target variable

## Approach
1. **Exploratory Data Analysis (EDA)**:
   - Checked for missing values (none found)
   - Examined class balance: bookings completed = ~15%, not completed = ~85%
   - Identified key categorical and numerical variables

2. **Data Preprocessing**:
   - One-hot encoded categorical variables
   - Kept numerical variables as is
   - Applied class balancing using `class_weight='balanced'` in the model

3. **Model Selection**:
   - Chose `RandomForestClassifier` to enable feature importance analysis
   - Combined preprocessing and model training using `Pipeline`

4. **Training and Evaluation**:
   - Split data into training (80%) and testing (20%) sets using stratified sampling
   - Evaluated model using accuracy, F1-score, and confusion matrix
   - Performed 5-fold cross-validation for robustness

5. **Feature Importance**:
   - Extracted feature importances from the trained RandomForest model
   - Identified top predictive features:
     1. Purchase Lead Time
     2. Trip Length
     3. Departure Hour
     4. Booking Origin (e.g., Australia, Malaysia)
     5. Flight Duration

## Results
- Model performance metrics (replace with your actual numbers):
  - Accuracy: XX%
  - F1 Score: XX%
  - Cross-validation F1 Average: XX%
- Timing-related variables (lead time, stay length, flight hour) are more influential than optional add-on services

## How to Run
1. Install dependencies:
```
pip install pandas numpy scikit-learn matplotlib seaborn
```
2. Place the dataset (`customer_booking.csv`) in the project directory
3. Run the Jupyter notebook or Python script containing the preprocessing, training, and evaluation code
4. Review model outputs and feature importance chart

## Files
- `customer_booking.csv`: Dataset
- `analysis.ipynb`: Data exploration, preprocessing, training, evaluation

## License
This project is for educational and demonstration purposes.
