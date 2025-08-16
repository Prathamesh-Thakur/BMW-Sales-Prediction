# :car: BMW Model Price Prediction
This project aims to predict the price of a BMW car model using a regression model trained on a Kaggle dataset. It includes an engineered feature called a popularity score and provides visualizations of price trends.

Check out the application here: [Link](https://bmw-sales-prediction-opnkyntqtaa447kzrohqu9.streamlit.app/)

# Properties
* Takes model name, region of sale, color, transmission type, fuel type and engine size to determine final price. 

* Uses a linear regression model to predict the price of the model.

# ⚙️ Engineered Feature 
* Popularity Score Calculation: This feature is a composite score that quantifies a car's popularity by analyzing the distribution of its key attributes—color, fuel type, transmission, and engine size.

* Weighted Metrics: The score incorporates scaled values of the car's price, sales volume, and mileage to provide a comprehensive measure of its market standing and demand.

* Temporal and Geographical Granularity: The popularity score is calculated with high precision, considering both the year of the data and the specific geographical region to capture localized trends.

* Dynamic and Context-Aware: Unlike static features, this score is dynamic; it provides a context-aware metric that reflects a car model's relevance and desirability within a specific time frame and location.

# :chart_with_upwards_trend: Data Visualization 
* Filtered Visualizations: The app generates a series of line charts, with each graph showing the price trend for a specific car attribute (e.g., mileage, engine size) based on the user's input.

* Sequential Filtering: Each subsequent graph is generated from a progressively filtered subset of the data. The filter is based on the attribute value selected from the previous graph, creating a cascading view.

* Attribute-Specific Trends: Each line chart isolates and displays the price trend exclusively for the attribute it represents, providing a clear, focused look at its relationship with price.

* Multi-Dimensional Analysis: The sequence of graphs allows users to perform a step-by-step, multi-dimensional analysis, exploring how price trends change as the dataset is narrowed down by different car characteristics.

# 📦 Dataset
The project uses a public dataset from Kaggle titled BMW Car Sales Classification [Dataset Link](https://www.kaggle.com/datasets/sumedh1507/bmw-car-sales-dataset). 

# 🚀 How to Run

* Clone the repository:

> git clone [repository-url]

> cd bmw-price-prediction

* Install dependencies:

> pip install -r requirements.txt

* Run the main script:

> streamlit run app.py

# 🛠️ Requirements
* Python 3.8+

* numpy

* pandas

* scikit-learn

# 🤝 Contributing
Contributions are welcome! Please open an issue or submit a pull request if you want to improve the project.
