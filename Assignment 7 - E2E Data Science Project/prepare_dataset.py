import numpy as np           
import pandas as pd           
import sys
import os
# Add the parent directory to the Python path
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
parent_dir = parent_dir + '/MAL1_Group_Assignemnt'
sys.path.append(parent_dir)
import google_drive as gd
from sklearn.model_selection import train_test_split

def prepare_alcohol_consumption_data():
    df_happy = gd.load_file('WHRAllYEars.xlsx')
    df_alcohol= gd.load_file('alcohol-consumption.csv')
    df_alcohol = df_alcohol.rename(columns = {'Total alcohol consumption per capita (liters of pure alcohol, projected estimates, 15+ years of age)': 'Alcohol_consumption'})
    df_alcohol = df_alcohol[['Entity', 'Year', 'Alcohol_consumption']]
    df_alcohol = df_alcohol.rename(columns = {'Entity': 'Country_Name'})
    dict_mappins = {'Turkey': 'Türkiye', 
                'Taiwan':  'Taiwan Province of China',
                'Palestine':'State of Palestine',
                'Congo': 'Congo (Brazzaville)',
                'Democratic Republic of Congo': 'Congo (Kinshasa)',
                'Hong Kong': 'Hong Kong S.A.R. of China',
                "Cote d'Ivoire": "Ivory Coast",
                'Somaliland': 'Somaliland region',
                }

    # Replace country names in df_alcohol using the mapping dictionary
    df_alcohol['Country_Name'] = df_alcohol['Country_Name'].replace(dict_mappins)
    # Get unique country names from both datasets
    target_countries = set(df_happy['Country_Name'].unique())
    # Filter the alcohol dataset to only include rows with countries in the target list
    df_alcohol_filtered = df_alcohol[df_alcohol['Country_Name'].isin(target_countries)]
    return df_alcohol_filtered

def prepare_fertility_data():
    df_happy = gd.load_file('WHRAllYEars.xlsx')
    df_fertility = gd.load_file('children_per_woman.xlsx')
    df_fertility['Fertility_Rate'] = pd.to_numeric(df_fertility['Fertility_Rate'], errors='coerce')
    df_fertility.drop(columns=['Code'], inplace=True)
    country_mapping = {
   "Turkey": "Türkiye",
    "Congo": "Congo (Brazzaville)",
    "Democratic Republic of Congo": "Congo (Kinshasa)",
    "Cote d'Ivoire": "Ivory Coast",
    "Hong Kong": "Hong Kong S.A.R. of China",
    "Palestine": "State of Palestine",
    "Taiwan": "Taiwan Province of China" }   
    df_fertility['Country_Name'] = df_fertility['Country_Name'].replace(country_mapping)
    countries_happy = set(df_happy['Country_Name'].unique())
    df_fertility = df_fertility[df_fertility['Country_Name'].isin(countries_happy)]
    return df_fertility

def prepare_migration_data():
    df_happy = gd.load_file('WHRAllYEars.xlsx')
    df_net_migration = gd.load_file('net_migration_by_country.xlsx')

    target_countries = df_happy['Country_Name'].unique()

    # Rename the countries in the migration dataset to match the target dataset
    df_net_migration['Country_Name'].replace({
        'Congo, Dem. Rep.': 'Congo (Brazzaville)',
        'Congo, Rep.': 'Congo (Kinshasa)',
        'Egypt, Arab Rep.': 'Egypt',
        'Gambia, The': 'Gambia',
        'Hong Kong SAR, China': 'Hong Kong S.A.R. of China',
        'Iran, Islamic Rep.': 'Iran',
        "Cote d'Ivoire": 'Ivory Coast',
        'Kyrgyz Republic': 'Kyrgyzstan',
        'Lao PDR': 'Laos',
        'Russian Federation': 'Russia',
        'Slovak Republic': 'Slovakia',
        'Somaliland':'Somaliland region',
        'Korea, Rep.': 'South Korea',
        'West Bank and Gaza': 'State of Palestine',
        'Syrian Arab Republic': 'Syria',
        'Turkiye': 'Türkiye',
        'Venezuela, RB': 'Venezuela',
        'Yemen, Rep.': 'Yemen'
    }, inplace=True)
    # Remove all the countries from the df_net_migration dataset that are not in the target list
    df_net_migration_filtered = df_net_migration[df_net_migration['Country_Name'].isin(target_countries)]
    return df_net_migration_filtered

def merge_hapiness_data_with_alcohol_consumption():
    df_happy = gd.load_file('WHRAllYEars.xlsx')
    df_alcohol = prepare_alcohol_consumption_data()
    df_merged = df_happy.merge(df_alcohol, on=['Country_Name', 'Year'], how='left')
    return df_merged

def merge_happiness_alcohol_fertility():
    df_happy_and_alcohol= merge_hapiness_data_with_alcohol_consumption()
    df_fertility = prepare_fertility_data()
    df_merged = df_happy_and_alcohol.merge(df_fertility, on=['Country_Name', 'Year'], how='left')
    return df_merged

def merge_happiness_alcohol_fertility_migration():
    df_happy_and_alcohol_fertility = merge_happiness_alcohol_fertility()
    df_migration = prepare_migration_data()
    df_merged = df_happy_and_alcohol_fertility.merge(df_migration, on=['Country_Name', 'Year'], how='left')
    return df_merged


def split_data_into_test_and_train():
    #DELETE rows:
    #Drop all rows that have 1 or 2 sample represented based on year count
    #This happens in the split data into test and train
    #-- Maldives, Oman, Belize, Bhutan, Suriname, Somaliland republic
    df = merge_happiness_alcohol_fertility_migration()
        # Count the occurrences of each Country_Name
    country_counts = df['Country_Name'].value_counts()

    # Filter out rows where the country appears only once or twice
    df = df[df['Country_Name'].isin(country_counts[country_counts > 2].index)]
        # Country Name is  string, we need to label encode that
    # Label Encoding
    df['Country_Encoded'] = df['Country_Name'].astype('category').cat.codes
    # Combine 'Year' and 'Country_Name' into a single string and set as index
    df['Year_Country'] = df['Year'].astype(str) + "_" + df['Country_Name']
    df = df.set_index('Year_Country')
    # Splitting the target and features
    target = df['Happines_Score']  # Extract the target column
    features = df.drop(columns=['Happines_Score'])  # Drop the target column from the features
    
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.2, random_state=42
    )
    return X_test, y_test, X_train, y_train, X_val, y_val



X_test, y_test, X_train, y_train, X_val, y_val = split_data_into_test_and_train()

