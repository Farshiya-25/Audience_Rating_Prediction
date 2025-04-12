# Rotten Tomatoes Audience Rating Prediction
This project aims to predict audience rating for movies based on their metadata using machine learning. The dataset used is the Rotten Tomatoes Movies3 dataset, containing detailed information such as movie titles, genres, runtime, release dates, and critic reviews.

## Dataset
The dataset for this project can be downloaded from Kaggle: [Dataset Link](https://www.kaggle.com/datasets/pranavvkathar/rotten-tomatoes-movies3-csv?select=Rotten_Tomatoes_Movies3.xls)

## Project Highlights

1. **Data Preprocessing**

   **a)Handle Missing Values / Imputation:**
     - Replace missing numerical values with the **mean** of the respective feature.
     - Replace missing textual and categorical values with `"Unknown"`.
   
   **b)Encode Categorical Features:**
     - For features with fewer unique values (e.g., `'rating'`, `'tomatometer_status'`), apply **One-Hot Encoding**.
     - For features with more unique values (e.g., `'genre'`, `'directors'`, etc.), apply **Frequency Encoding**.
   
   **c)Apply TF-IDF Vectorization for textual Feature:**
     - Transform the 'movie_info' text column using **TF-IDF vectorization** to convert text into numerical features, removing English      stopwords.

2. **Feature Extraction:**

   **a)Sentiment Score:**
     - Create a new feature **"sentiment_score"** derived from `'critics_consensus'` using the **TextBlob().sentiment.polarity**.

   **b)Date Features:**
     - Extract the following date-related features from `'in_theaters_date'` and `'on_streaming_date'`:
       - `'release_year'`, `'streaming_days'`** (the difference between `'in_theaters_date'` and `'on_streaming_date'`)

3. **Feature Selection Using Correlation Analysis:**

   - Calculate **Pearson correlation** of each numerical and categorical feature with the target column **`'audience_rating'`**.
   - Visualize the correlation matrix using a **bar chart**.
   
   - **Selected Features:**
     - Features with medium to high correlation with the target (`'audience_rating'`) are selected. These include:
        `'runtime_in_minutes'`, `'tomatometer_rating'`, `'tomatometer_count'`
       `'directors_Frequency'`, `'tomatometer_status_Certified Fresh'`, `'tomatometer_status_Fresh'`, `'tomatometer_status_Rotten'`
        `'sentiment_score'`, `'in_theaters_year'`,  `'streaming_days'`
       - Additionally, the  `movie_info` and `critics_consensus` columns are selected.

 4. **Normalization:**

   - The following columns are **normalized**:
     `'runtime_in_minutes'`, `'tomatometer_rating'`, `'tomatometer_count'`, `'directors_Frequency'`, `'in_theaters_weekday'`, `'days_between'`.

5. **Model Training:**

   **a)Data Split:**
     - Split the dataset into **training** and **test** sets.
   
   **b)Model Selection:**
     - Three models are compared and evaluated:
       - **Linear Regression**
       - **Random Forest Regression**
       - **XGBoost Regression**
       - **Gradient Boosting Regressor**
     - The model with the **least RMSE** and the **highest R-squared (R²)** value is selected, which is **XGBRegressor**.
   
   **c)Hyperparameter Tuning:**
     - Use **Grid Search** to tune hyperparameters and find the best set of hyperparameters for the **XGBRegressor** model.

 6. **Model Pipeline:**

   - Design the **model pipeline** using the best model and optimal hyperparameters.
   - Train the pipeline using the **training** data.
   - Evaluate the model on the **test set** and calculate the performance metrics.



### **Accuracy Calculation:**

The accuracy can be calculated using the **Mean Absolute Percentage Error (MAPE)** formula:

$$
\text{MAPE} = \frac{1}{n} \sum_{i=1}^{n} \left|\frac{\hat{yPred}_i - yActual_i}{yActual_i}\right| \times 100
$$


The **accuracy** is then calculated as:

$$
\text{Accuracy} = 100 - \text{MAPE}
$$

In this case, the accuracy is **75.8%**.
