import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import f_oneway
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# Load dataset
df = pd.read_csv('car_data.csv')
print("Dataset Head:")
print(df.head())

# Summary Statistics
print("\nSummary Statistics:")
print(df.describe())

# Check for Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

# Drop irrelevant columns (e.g., Car_Name)
df = df.drop('Car_Name', axis=1)

# Convert 'Year' to 'Age'
current_year = 2024
df['Age'] = current_year - df['Year']
df = df.drop('Year', axis=1)

# Encode categorical variables (Label Encoding)
le = LabelEncoder()
df['Fuel_Type'] = le.fit_transform(df['Fuel_Type'])  # Petrol: 2, Diesel: 1, CNG: 0
df['Seller_Type'] = le.fit_transform(df['Seller_Type'])  # Dealer: 1, Individual: 0
df['Transmission'] = le.fit_transform(df['Transmission'])  # Manual: 1, Automatic: 0

# Save encoders for later use 
joblib.dump(le, 'label_encoder.pkl')

#EDA
# 1. Distribution of Selling Price
plt.figure(figsize=(8, 6))
sns.histplot(df['Selling_Price'], kde=True, color='blue')
plt.title('Distribution of Selling Price')
plt.xlabel('Selling Price')
plt.ylabel('Frequency')
plt.show()

# 2. Correlation Heatmap
plt.figure(figsize=(10, 8))
corr = df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

# 3. Pairplot for Numerical Features
sns.pairplot(df[['Selling_Price', 'Present_Price', 'Kms_Driven', 'Age']])
plt.suptitle('Pairplot of Numerical Features', y=1.02)
plt.show()

# 4. Boxplot for Categorical Features
plt.figure(figsize=(12, 6))
sns.boxplot(x='Fuel_Type', y='Selling_Price', data=df)
plt.title('Selling Price vs Fuel Type')
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(x='Seller_Type', y='Selling_Price', data=df)
plt.title('Selling Price vs Seller Type')
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(x='Transmission', y='Selling_Price', data=df)
plt.title('Selling Price vs Transmission')
plt.show()

# Function to perform ANOVA
def perform_anova(data, categorical_feature, target_feature):
    categories = data[categorical_feature].unique()
    groups = [data[data[categorical_feature] == category][target_feature] for category in categories]
    f_stat, p_value = f_oneway(*groups)
    print(f"ANOVA for {categorical_feature}:")
    print(f"F-statistic: {f_stat:.2f}, p-value: {p_value:.4f}")

# Perform ANOVA for each categorical feature
perform_anova(df, 'Fuel_Type', 'Selling_Price')
perform_anova(df, 'Seller_Type', 'Selling_Price')
perform_anova(df, 'Transmission', 'Selling_Price')

# Split data into features (X) and target (y)
X = df.drop('Selling_Price', axis=1)
y = df['Selling_Price']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, 'models/car_price_predictor.pkl')

# Evaluate the model
y_pred = model.predict(X_test)
print(f'Mean Absolute Error: {mean_absolute_error(y_test, y_pred)}')
print(f'R2 Score: {r2_score(y_test, y_pred)}')