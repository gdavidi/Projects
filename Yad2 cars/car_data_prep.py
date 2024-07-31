import re
import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz
from datetime import datetime, timedelta
# ------------------------------------------------------------ Functions ------------------------------------------------------------

# Convert excel serial date to datetime object function
def excel_date_to_date(excel__serial_date):
    return datetime(1899, 12, 30) + timedelta(days=excel__serial_date) # Excel serial date starts from 30/12/1899

# Extract the year and clean the text function
def extract_year_and_clean_text(text):
    pattern = r'\s*\(?(\d{4})\)?,?\s*'  # Extract the year from the text 
    match = re.search(pattern, text) # Search for the pattern in the text
    if match: # If a match is found
        year = match.group(1) # Extract the year
        cleaned_text = re.sub(pattern, '', text) # Remove the year from the text
        return cleaned_text, year # Return the cleaned text and the year
    return text, None # If no match is found, return the original text and None

# Finding the similarity between pairs of strings function
def find_similar_strings(strings, threshold=90):
    similar_pairs = [] # List to store the similar pairs
    for i in range(len(strings)): # Iterate over the strings
        for j in range(i + 1, len(strings)): # Iterate over the strings after the current string
            similarity = fuzz.ratio(strings[i], strings[j]) # Calculate the similarity between the strings
            if similarity >= threshold: # If the similarity is above the threshold
                similar_pairs.append((strings[i], strings[j], similarity)) # Add the pair to the list
    return similar_pairs

# Print similar string values in a series function
def print_similar_string_values_in_series(series, threshold=90): 
    strings = list(series.unique()) # Get the unique values in the series
    similar_strings = find_similar_strings(strings, threshold) # Find similar strings
    for pair in similar_strings: # Iterate over the similar pairs
        print(f"Pair: {pair[0]}\nand {pair[1]}\nwith similarity {pair[2]}%\n\n") # Print the pair
        
# Description cleaning function
def clean_description(desc):
    desc = re.sub(r"[^א-ת\s]", "", desc) # Remove non-Hebrew characters
    tokens = desc.split() # Tokenize the description
    return tokens

# Identify common meaning words function
def select_common_word(tokens):
    common_meaning_words = {'מטופל', 'חדש', 'שמור', 'מתוחזק', 'תוספות', 'מצוין', 'מעולה', 'נהדר','שמורה','ללא','מיבוא','חסכוני','מצויין','טיפול'}
    for word in tokens: # Iterate over the words
        if word in common_meaning_words: # If the word is in the common meaning words
            return word # Return the word
    return 'אחר'
        
# Cleaning the 'Description' column, selecting common meaning words, and replacing some words with synonyms (/similar words)
def description_features(df):
    df['Description'] = df['Description'].apply(clean_description) # Clean the description
    df['Description'] = df['Description'].apply(select_common_word) # Select common meaning words
    df['Description'] = df['Description'].replace('שמורה','שמור')  
    df['Description'] = df['Description'].replace('מעולה','מצוין')
    df['Description'] = df['Description'].replace('נהדר','מצוין')
    df['Description'] = df['Description'].replace('טיפול','מטופל')
    df['Description'] = df['Description'].replace('מתוחזק','מטופל')
    df['Description'] = df['Description'].replace('מצויין','מצוין')
      
# Feature engineering/handling for dates
def to_date(df, threshold=0.95): 
    date_regex_pattern = r'^\d{2}/\d{2}/\d{4}$' # Date regex pattern
    try:
        non_matching_rows_Cre_date = df[~(df['Cre_date'].str.contains(date_regex_pattern, regex=True))] # Find rows with non-matching dates
        df.loc[non_matching_rows_Cre_date.index, 'Cre_date'] = df.loc[non_matching_rows_Cre_date.index, 'Cre_date'].astype('Int64').apply(excel_date_to_date).dt.strftime('%d/%m/%Y') # Convert the dates to the correct format
        df['Cre_date'] = pd.to_datetime(df['Cre_date'], format="%d/%m/%Y") # Convert the dates to datetime format
    except: 
        pass 
    finally: 
        if df.shape[0] > 1: # If the dataframe has more than one row, it means it's the training data (and not the user input data)
            non_matching_rows_Repub_date = df[~(df['Repub_date'].str.contains(date_regex_pattern, regex=True))] # Find rows with non-matching dates
            df.loc[non_matching_rows_Repub_date.index, 'Repub_date'] = df.loc[non_matching_rows_Repub_date.index, 'Repub_date'].astype('Int64').apply(excel_date_to_date).dt.strftime('%d/%m/%Y') # Convert the dates to the correct format
        df['Repub_date'] = pd.to_datetime(df['Repub_date'], format="%d/%m/%Y") # Convert the dates to datetime format

# Get the season of a date function
def get_season(date): 
    year = date.year # Get the year
    seasons = {
        'winter': pd.date_range(start=f'01/12/{year}', end=f'31/12/{year}'), # Winter starts from 1st December to 28th February
        'winter': pd.date_range(start=f'01/01/{year}', end=f'28/02/{year}'), # Leap year, until 28th February
        'spring': pd.date_range(start=f'01/03/{year}', end=f'31/05/{year}'), # Spring starts from 1st March to 31st May
        'summer': pd.date_range(start=f'01/06/{year}', end=f'31/08/{year}'), # Summer starts from 1st June to 31st August
        'autumn': pd.date_range(start=f'01/09/{year}', end=f'30/11/{year}'), # Autumn starts from 1st September to 30th November
    }
    for season in seasons: # Iterate over the seasons
        if date in seasons[season]: # If the date is in the season
            return season # Attach the season to the date
    return None      

# Replacing similar meaning words       
def ownership_features(df):
    df['Prev_ownership'] = df['Prev_ownership'].replace('השכרה', 'ליסינג') 
    df['Prev_ownership'] = df['Prev_ownership'].replace('חברה', 'ליסינג')
    df['Prev_ownership'] = df['Prev_ownership'].replace('ממשלתי', 'ליסינג')
    df['Prev_ownership'] = df['Prev_ownership'].replace('לא מוגדר', 'פרטית')
    df['Prev_ownership'] = df['Prev_ownership'].replace('אחר', 'פרטית')
    df['Prev_ownership'] = df['Prev_ownership'].fillna(df['Prev_ownership'].mode()[0])

# ------------------------------------------------------------ Main ------------------------------------------------------------

def prepare_data(df):
    
    # Remove duplicates
    df = df.drop_duplicates(keep='first').reset_index(drop=True)

    # Remove white spaces from all string columns
    string_columns = df.select_dtypes(include=['object']).columns
    for col in string_columns: 
        df[col] = df[col].str.strip().str.replace(r'[\r\n\t]', '', regex=True) 

    # Remove columns that...( The 'if's are used to avoid errors if the column doesn't exist in the dataframe)
    if 'Test' in df.columns: # Removed due to having too many missing values
        df = df.drop(['Test'], axis=1)
    if 'Supply_score' in df.columns: # Removed due to having too many missing values
        df = df.drop(['Supply_score'], axis=1)
    if 'Curr_ownership' in df.columns: # Removed because it shares 0.88% of the same values with Curr_ownership
        df = df.drop(['Curr_ownership'], axis=1)
    try:
        df = df.dropna(how='any', subset=['Year', 'model', 'manufactor', 'Gear']) # Removed rows with missing values in these columns because they are important for the model
    except:
        pass
    # The following columns were removed due to not having any 'logical' meaning correlation with the price:
    if 'Color' in df.columns:
        df = df.drop(['Color'], axis=1)
    if 'Area' in df.columns:
        df = df.drop(['Area'], axis=1)
    if 'City' in df.columns:
        df = df.drop(['City'], axis=1)
    if 'Pic_num' in df.columns:
        df = df.drop(['Pic_num'], axis=1) 

    df['manufactor'] = df['manufactor'].replace('Lexsus', 'לקסוס')  
    df['Gear'] = df['Gear'].replace('אוטומטי', 'אוטומטית')

    df['capacity_Engine'] = df['capacity_Engine'].str.replace(',','').astype('Int64')
    df['capacity_Engine'] = df['capacity_Engine'].fillna(df['capacity_Engine'].median().astype('int64')).astype('int64')
        
    df['Km'] = df['Km'].str.replace(',','').astype('Int64')
    df['Km'] = df['Km'].fillna(df['Km'].median().astype('int64')).astype('int64')
    df = df[df['Km'] < 400000].reset_index(drop=True) # Removing outliers (Car that drove more than 400,000 km shouldn't drive again! All the more so sell it)

    df['Engine_type'] = df['Engine_type'].replace('היבריד','היברידי')
    df['Engine_type'] = df['Engine_type'].replace('לא מוגדר', np.nan)
    df['Engine_type'] = df['Engine_type'].fillna(df['Engine_type'].mode()[0])

    to_date(df) # Cleaning invalid dates and changing all values type to datetime 
    ownership_features(df) # Explaination in the function itself (at the beggining of the code)
    description_features(df) # Explaination in the function itself (at the beggining of the code)

    # Creating new column named 'Season' that contains the season of each date in 'Repub_date' column, thinking it might hold some correlation with the price. In any case, it's better than having useless dates
    df['Season'] = df['Repub_date'].apply(get_season)
    df['Season'] = df['Season'].fillna(df['Season'].mode()[0])

    if 'Cre_date' in df.columns: # Removed due to having low predictive power score with Price, and creating Season column instead (feature engineering)
        df = df.drop(['Cre_date'], axis=1)
    if 'Repub_date' in df.columns: # Removed due to having low predictive power score with Price, and creating Season column instead (feature engineering)
        df = df.drop(['Repub_date'], axis=1)

    # Cleaning the rows that contain years in the 'model' column by removing the year, and keeping only the model name
    car_models = list(df['model'].unique())
    elements_to_remove = ["GT3000", "5008", "2008"]
    for element in elements_to_remove: # Removing elements that are not car models
        if element in car_models: # If the element is in the car models list
            car_models.remove(element) # Remove the element
    df['model'] = df['model'].apply(lambda model_name: extract_year_and_clean_text(model_name)[0] if extract_year_and_clean_text(model_name)[1] and (model_name in car_models) else model_name) # Clean the model names

    categorical_features = ['manufactor', 'model', 'Gear', 'Engine_type', 'Prev_ownership', 'Description', 'Season']
    numerical_features = ['Year', 'Hand', 'capacity_Engine', 'Km']

    if df.shape[0] > 1: # If the dataframe has more than one row, it means it's the training data (and not the user input data)
    # It is required to identify between the training data and the user input data before using dummies, because if the data is the user input it only has one row.
    # If all values in a categorical column are the same, get_dummies does not create any dummy variables for that column when drop_first=True is used.
        df = pd.get_dummies(df, columns=categorical_features, drop_first=True)
    else: 
        df = pd.get_dummies(df, columns=categorical_features)
        
    return df