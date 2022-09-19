# PREDICTIVE ANALYSIS ON HOTEL BOOKING CANCELLATIONS

# **Machine Learning Pipeline**

1. Problems Framing and Data Understanding
2. Exploratory Data Analysis (EDA)
3. Feature Engineering
4. Model Training  
5. Evaluation, Model Selection, and Model Tuning  
6. Conclusion and Recommendation

> ## *Context*

Accurate information on room availability can prevent the hotel from various losses, such as profit loss caused by guests who cancel reservations at the last minute.

Guests who cancel their reservations, either notifying the hotel in advance (e.g., contacting themselves or being contacted by the hotel) or not, will cause vacancies in the rooms they should be staying in. These rooms can actually still be rented again to other guests, but it will be more difficult to decide immediately which rooms can be republished if the news of the cancellation is received at the last minute/ few days before the check-in date.

The cancellation of bookings impacts a hotel on various fronts:

- Loss of resources (revenue) when the hotel cannot resell the room.
- Additional costs of distribution channels by increasing commissions or paying for publicity to help sell these rooms.
- Lowering prices at the last minute so the hotel can resell a room, resulting in reducing the profit margin.

Therefore, predicting reservations that can be cancelled and preventing these cancellations will create a surplus value for the institutions.

In this scenario, a hotel business owner in Bali asked for help to solve their problem, which has a 37% cancellation rate. As a data scientist, I would like to provide solutions by identifying problems, analysing data to generate insights, creating machine learning models with algorithms that can predict as accurately as possible on the cancellation, and providing recommendations based on the overall results I'll receive. 

Here is the definition of each target for this case:

    0 = Not cancelled guests 
    Guests who don't cancel the reservation defined as one who shows up at the hotel.

    1 = Cancelled guests
    Guests who cancel the reservation defined as one who cancels days before arrival date or at the last minute.


> ## *Business Problem*

**Problem**:  
There are two important points that need to be given attention before formulating the problem, such as goals and values. Our goal here is to predict the booking cancellations as accurately as possible in order to achieve values in which to mitigate revenue loss derived from booking cancellations and the risks associated with overbooking.

**Machine Learning Task**

Here I will make a prediction on cancellations using machine learning model with the Supervised Learning method for Binary Classification. There is already a ground truth, or a marker, whether someone cancels their booking or not. In addition to providing predictions on booking cancellations, the stakeholders also hope to find out the factors/variables that affect guests' decisions to cancel, so that they can make a good strategy in anticipating high cancellation rates.

**Metric Analysis**  

So, the model we are after is a model that provides accurate predictions in the positive class (cancelled guests) with a higher recall score to handle losing revenue and unutilised capacity. However, we need to make sure that the precision score has a good measure to avoid the risk of overbooking. So we have to balance between precision and recall of 1 positive class. So the main metric of prediction will be **f1-score**, but we will also pay attention to the **recall** score and ensure that it is greater than **precision**. In addition, the purpose of using the f1-score is one method to overcome the imbalance of data in the positive class (Guest Cancel) and the negative class (Guest Show-up).

> ## *Data Summary*

- We have 83,573 rows/observations and 11 features/columns.
- The only feature that has missing values in the train set is in column `country`. It takes 0.42% out of 83,573 observations.
- We have 5 categorical data such as `country`, `market_segment`, `deposit_type`, `customer_type`, and `reserved_room_type`.
- The following are the numerical features in our data: `previous_cancellations`, `booking_changes`, `days_in_waiting_list`, `required_car_parking_spaces`, and `total_of_special_requests`.
- The feature that will be considered as our target is `is_canceled`, where 0 indicates `not canceled` and 1 as `canceled`.*


> ## *Feature Selection by EDA*
Based on the EDA analysis above, we came up to removing 2 numerical features for the modeling process. That now leaves us with 8 features as follows:

1. total_of_special_requests
2. required_car_parking_spaces
3. booking_changes
4. country
5. market_segment
6. deposit_type
7. customer_type
8. reserved_room_type

> ## **Model Selection**

I trained 3 different tree-based models: Random Forest, AdaBoost Classifier, and XGBoost Classifier.

These three models are SOTA (state-of-the-art) models because they are a combination of weak-learners that are bagged or boosted into one model that is able to provide good accuracy.


