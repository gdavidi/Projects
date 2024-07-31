import pandas as pd
from flask import Flask, request, render_template
import pickle
import os
import json
from car_data_prep import prepare_data

app = Flask(__name__) # Initialize the Flask app

# Load the trained model
elnet_model = pickle.load(open('trained_model.pkl', 'rb'))

# Load the list of columns used during training (to ensure all columns are present and in the correct order)
columns_list = json.load(open('columns_list.json', 'r'))

@app.route('/') # Home page route
def home(): 
    return render_template('index.html', prediction_text='') # Render the index.html template

@app.route('/predict',methods=['POST']) # Prediction route
def predict():
    features_names = ['manufactor', 'Year', 'model', 'Hand', 'Gear', 'capacity_Engine', 'Engine_type', 'Prev_ownership', 'Curr_ownership', 'Area', 'City', 'Pic_num', 'Cre_date', 'Repub_date', 'Description', 'Color', 'Km', 'Test']
    features_values = request.form.getlist('feature')
    
    features_values = [
        features_values[0],                                 # manufactor (str)
        int(features_values[1]),                            # Year (int)
        features_values[2],                                 # model (str)
        int(features_values[3]),                            # Hand (int)
        features_values[4],                                 # Gear (str)
        features_values[5],                                 # capacity_Engine (str, will be converted to int in prepare_data)
        features_values[6],                                 # Engine_type (str)
        features_values[7],                                 # Prev_ownership (str)
        features_values[8] if features_values[8] else "", # Curr_ownership (str)
        features_values[9] if features_values[9] else "", # Area (str)
        features_values[10] if features_values[10] else "", # City (str)
        int(features_values[11]) if features_values[11] else 0, # Pic_num (int)
        features_values[12] if features_values[12] else "", # Cre_date (str, will be converted to date in prepare_data)
        features_values[13],                                # Repub_date (str, will be converted to date in prepare_data)
        features_values[14] if features_values[14] else "", # Description (str)
        features_values[15] if features_values[15] else "", # Color (str)
        features_values[16],                                # Km (str, will be converted to int in prepare_data)
        features_values[17] if features_values[17] else ""  # Test (str, will be handled in prepare_data)
    ]
    
    # Create a dataframe from the user input
    features_dataframe = pd.DataFrame({name: [value] for name, value in zip(features_names, features_values)}) # Create a dataframe from the user input
    
    # Prepare the data for prediction
    processed_df = prepare_data(features_dataframe)

    # Ensure all necessary columns are present in the dataframe from the user input, to be able to fit the model on it. 
    missing_cols = [col for col in columns_list if col not in processed_df.columns] # Find the missing columns
    missing_df = pd.DataFrame(0, index=processed_df.index, columns=missing_cols) # Create a dataframe with the missing columns and fill it with 0
    processed_df = pd.concat([processed_df, missing_df], axis=1) # Concatenate the missing columns to the processed_df dataframe
    
    # Reorder columns to match the training dataset
    processed_df = processed_df[columns_list] 
    
    # Make a prediction
    prediction = elnet_model.predict(processed_df)[0] 
    # Format the output
    output_text =  f"{int(prediction)}₪ :מחיר מוערך" 
    
    # Render the index.html template with the prediction
    return render_template('index.html', prediction_text=output_text) 

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000)) 
    app.run(host='0.0.0.0', port=port,debug=True)