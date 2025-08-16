# Library imports
import numpy as np
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle

# Read the dataset file
df = pd.read_csv("BMW_Car_Sales_Classification.csv")

# Initialize the Popularity_Score column
df['Popularity_Score'] = np.nan

# Convert relevant numeric columns to float type
df['Price_USD'] = df['Price_USD'].astype(float, errors = 'ignore')
df['Sales_Volume'] = df['Sales_Volume'].astype(float, errors = 'ignore')
df['Mileage_KM'] = df['Mileage_KM'].astype(float, errors = 'ignore')

# Function to calculate the popularity score
def find_pop_score(level2):
    # Finds the proportion of unique values for each feature to be used in the model.
    # normalize = True converts the counts to proportions between 0 and 1
    transmission_props = level2.Transmission.value_counts(normalize = True)
    fuel_props = level2.Fuel_Type.value_counts(normalize = True)
    color_props = level2.Color.value_counts(normalize = True)
    engine_props = level2.Engine_Size_L.value_counts(normalize = True)

    # Initialize scaler object
    scaler = MinMaxScaler()

    # Scale the columns which we are going to use
    level2.loc[:, 'Price_USD'] = scaler.fit_transform(level2[['Price_USD']])
    level2.loc[:, 'Sales_Volume'] = scaler.fit_transform(level2[['Sales_Volume']])
    level2.loc[:, 'Mileage_KM'] = scaler.fit_transform(level2[['Mileage_KM']])

    # Iterate through the dataframe
    for index, row in level2.iterrows():
        # Find the proportions for each feature inputted by the user
        color = color_props[row['Color']]
        fuel_type = fuel_props[row['Fuel_Type']]
        transmission = transmission_props[row['Transmission']]
        engine_size = engine_props[row['Engine_Size_L']]
        mileage = row['Mileage_KM']

        # Calculate the popularity score
        pop_score = color * fuel_type * transmission * engine_size * row['Sales_Volume'] * row['Price_USD'] * mileage

        # Populate the score in the dataframe
        level2.loc[index, 'Popularity_Score'] = pop_score

    # Scale the popularity score and return the series
    level2.loc[:, 'Popularity_Score'] = scaler.fit_transform(level2[['Popularity_Score']])

    return level2[['Popularity_Score']]

# Populate the scores in the original dataframe
def populate_pop_scores():
    # Find the unique years
    years = df['Year'].unique()

    for year in years:
        # Slice the dataframe for the specific year
        level1 = df[df['Year'] == year]

        # Find all unique regions of sale for the year
        regions = level1['Region'].unique()

        for region in regions:
            # Slice the dataframe for the specific region
            level2 = level1[level1['Region'] == region]

            # Get the popularity scores for all the models for the region
            updated_level2 = find_pop_score(level2)

            # Populate the original dataframe
            for index in updated_level2.index:
                df.loc[index, 'Popularity_Score'] = updated_level2.loc[index, 'Popularity_Score']


# Function call to populate the scores
populate_pop_scores()

# Splitting for training and evaluation of the model
X = df[['Model', 'Region', 'Color', 'Transmission', 'Fuel_Type', 'Engine_Size_L', 'Popularity_Score']]
y = df['Price_USD'] / 100000

# Label encoders for all categorical values used in training
le_model = LabelEncoder()
le_region = LabelEncoder()
le_color = LabelEncoder()
le_fuel = LabelEncoder()
le_transmission = LabelEncoder()

le_model.fit(X['Model'])
le_region.fit(X['Region'])
le_color.fit(X['Color'])
le_fuel.fit(X['Fuel_Type'])
le_transmission.fit(X['Transmission'])

X.loc[:, 'Model'] = le_model.transform(X['Model'])
X.loc[:, 'Region'] = le_region.transform(X['Region'])
X.loc[:, 'Color'] = le_color.transform(X['Color'])
X.loc[:, 'Fuel_Type'] = le_fuel.transform(X['Fuel_Type'])
X.loc[:, 'Transmission'] = le_transmission.transform(X['Transmission'])

# Training and testing the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

model = LinearRegression()
model.fit(X_train, y_train)

# Save model weights
pickle.dump(model, open('model_weights.pickle', 'wb'))

preds = model.predict(X_test)

print("MAE of the model:", mean_squared_error(preds, y_test))
print("MSE of the model:", mean_absolute_error(preds, y_test))
