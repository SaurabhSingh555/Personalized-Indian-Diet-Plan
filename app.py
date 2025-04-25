import pickle
import pandas as pd
from flask import Flask, render_template, request, jsonify
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import os
from dotenv import load_dotenv
load_dotenv()



# Load the trained model and encoders
from joblib import dump, load

# Save model
dump(classifier, 'models/classifier.joblib')

# Load model
classifier = load('models/classifier.joblib')

# Diet information database (could be moved to a proper database)
DIET_TIPS = {
    'Weight Loss': [
        "Eat more fiber-rich foods like whole grains and vegetables",
        "Include protein in every meal to stay full longer",
        "Limit sugar and processed foods",
        "Drink plenty of water throughout the day",
        "Practice portion control"
    ],
    'Weight Gain': [
        "Eat calorie-dense foods like nuts and dried fruits",
        "Have frequent meals (5-6 per day)",
        "Include healthy fats like ghee and nuts",
        "Combine strength training with your diet",
        "Consider protein shakes if needed"
    ],
    'Maintain Weight': [
        "Balance your macronutrients (carbs, proteins, fats)",
        "Stay active with regular exercise",
        "Monitor your weight weekly",
        "Eat seasonal and varied foods",
        "Practice mindful eating"
    ]
}

REGIONAL_SPECIALTIES = {
    'North': ["Roti", "Dal Makhani", "Paneer dishes", "Paratha", "Lassi"],
    'South': ["Dosa", "Idli", "Sambar", "Rasam", "Coconut chutney"],
    'East': ["Macher Jhol", "Litti Chokha", "Pitha", "Rosogolla", "Misti Doi"],
    'West': ["Dhokla", "Thepla", "Pav Bhaji", "Vada Pav", "Shrikhand"]
}

# Home page
@app.route('/')
def home():
    return render_template('index.html')

# Recommendation API endpoint
@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        # Get form data
        age = int(request.form['age'])
        gender = request.form['gender']
        height = float(request.form['height'])
        weight = float(request.form['weight'])
        diet_type = request.form['diet_type']
        region = request.form['region']
        goal = request.form['goal']
        activity_level = request.form['activity_level']
        
        # Calculate BMI
        bmi = round(weight / ((height/100) ** 2), 1)
        
        # Encode categorical features
        gender_encoded = label_encoders['Gender'].transform([gender])[0]
        diet_type_encoded = label_encoders['DietType'].transform([diet_type])[0]
        region_encoded = label_encoders['Region'].transform([region])[0]
        goal_encoded = label_encoders['Goal'].transform([goal])[0]
        activity_level_encoded = label_encoders['ActivityLevel'].transform([activity_level])[0]
        
        # Prepare input data for prediction
        input_data = pd.DataFrame([[age, gender_encoded, bmi, diet_type_encoded, 
                                  region_encoded, goal_encoded, activity_level_encoded]],
                                columns=['Age', 'Gender', 'BMI', 'DietType', 
                                         'Region', 'Goal', 'ActivityLevel'])
        
        # Make prediction
        meal_plan = classifier.predict(input_data)[0]
        
        # Get additional recommendations
        bmi_category = get_bmi_category(bmi)
        diet_tips = DIET_TIPS.get(goal, [])
        regional_foods = REGIONAL_SPECIALTIES.get(region, [])
        
        return render_template('recommendation.html',
                             meal_plan=meal_plan,
                             bmi=bmi,
                             bmi_category=bmi_category,
                             diet_tips=diet_tips,
                             regional_foods=regional_foods,
                             goal=goal,
                             region=region)
    
    except Exception as e:
        return render_template('index.html', error=str(e))

# BMI category helper function
def get_bmi_category(bmi):
    if bmi < 18.5:
        return "Underweight"
    elif 18.5 <= bmi < 25:
        return "Normal weight"
    elif 25 <= bmi < 30:
        return "Overweight"
    else:
        return "Obese"

# About page
@app.route('/about')
def about():
    return render_template('about.html')

# Contact page
@app.route('/contact')
def contact():
    return render_template('contact.html')

if __name__ == '__main__':
    load_dotenv()  # Load environment variables
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)