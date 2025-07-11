# app.py

import streamlit as st
import joblib  # ✅ use joblib instead of pickle
import difflib  # ✅ for fuzzy matching

# Load saved model and vectorizer
saved = joblib.load('nutrition_model.pkl')

vectorizer = saved['vectorizer']
models = saved['models']
dish_list = saved['dishes']

# Fuzzy matcher function using difflib
def find_best_match(name: str, choices: list, score_cutoff: float = 0.6):
    matches = difflib.get_close_matches(name.lower(), choices, n=1, cutoff=score_cutoff)
    if matches:
        return matches[0].title(), 100  # Returning 100 as dummy score
    else:
        return None, 0

# Prediction function
def predict_nutrition(dish_name: str):
    best_match, score = find_best_match(dish_name, dish_list)
    if best_match is None:
        return None
    vec = vectorizer.transform([best_match])
    nutrition = {
        'Dish': best_match,
        'Calories (kcal)': round(models['calories'].predict(vec)[0], 1),
        'Carbs (g)': round(models['carbs'].predict(vec)[0], 1),
        'Protein (g)': round(models['protein'].predict(vec)[0], 1),
        'Fat (g)': round(models['fat'].predict(vec)[0], 1)
    }
    return nutrition

# Streamlit UI
st.set_page_config(page_title="🍛 Indian Food Nutrition Predictor", page_icon="🥗", layout="centered")

st.title("🥘 Indian Food Nutrition Predictor")
st.write("Enter the name of an Indian dish to predict its nutritional values (Calories, Carbs, Protein, Fat).")

# User input
dish_name = st.text_input("🍴 Enter Dish Name", placeholder="e.g. Pav Bhaji")

if st.button("Predict Nutrition"):
    if not dish_name.strip():
        st.warning("⚠️ Please enter a dish name.")
    else:
        result = predict_nutrition(dish_name)
        if result:
            st.success(f"✅ Nutrition for {result['Dish']}")
            st.write(f"🔥 **Calories**: {result['Calories (kcal)']} kcal")
            st.write(f"🍚 **Carbs**: {result['Carbs (g)']} g")
            st.write(f"🥩 **Protein**: {result['Protein (g)']} g")
            st.write(f"🧈 **Fat**: {result['Fat (g)']} g")
        else:
            st.error("❌ Dish not found in database. Try a different name.")
