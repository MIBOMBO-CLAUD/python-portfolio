import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load the dataset normally
df = pd.read_csv("house_data.csv")

print(df.head())
print(df.columns.tolist())

# Select input features and target
X = df[["Square_Feet", "Bedrooms", "Age"]]
y = df["Price"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Show results
results = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": predictions
})

print("\nPrediction Results:")
print(results)