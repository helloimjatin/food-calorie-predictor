# calorie_predictor.py

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib  # ✅ use joblib instead of pickle
import numpy as np  # ✅ for RMSE calculation

# Load dataset
data = pd.read_csv(r"C:\Users\ACER\Downloads\indian_dishes_dataset.csv")

# Preprocess
X = data['Dish Name']
y_calories = data['Calories (kcal)']
y_carbs = data['Carbs (g)']
y_protein = data['Protein (g)']
y_fat = data['Fat (g)']

# Vectorize dish names
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

# Train models
model_calories = LinearRegression()
model_carbs = LinearRegression()
model_protein = LinearRegression()
model_fat = LinearRegression()

model_calories.fit(X_vec, y_calories)
model_carbs.fit(X_vec, y_carbs)
model_protein.fit(X_vec, y_protein)
model_fat.fit(X_vec, y_fat)

# Test accuracy (optional)
X_train, X_test, y_train, y_test = train_test_split(X_vec, y_calories, test_size=0.2, random_state=42)
y_pred = model_calories.predict(X_test)

# Compute RMSE manually (compatible with older sklearn)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"✅ Calories Model RMSE: {rmse:.2f}")

# Save models and vectorizer with joblib ✅
joblib.dump({
    'vectorizer': vectorizer,
    'models': {
        'calories': model_calories,
        'carbs': model_carbs,
        'protein': model_protein,
        'fat': model_fat
    },
    'dishes': list(X.str.lower())
}, 'nutrition_model.pkl')

print("🎉 Model and vectorizer saved as nutrition_model.pkl")
