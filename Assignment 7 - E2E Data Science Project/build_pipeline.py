import numpy as np           
import pandas as pd           
import sys
import os
# Add the parent directory to the Python path
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
parent_dir = parent_dir + '/MAL1_Group_Assignemnt'
sys.path.append(parent_dir)
from sklearn.model_selection import train_test_split
from prepare_dataset import split_data_into_test_and_train
from sklearn import linear_model
from sklearn import (datasets, decomposition, ensemble, 
                     metrics, model_selection, preprocessing)
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from prepare_dataset import merge_happiness_alcohol_fertility_migration

def build_pipeline():
    #DELETE rows:
    #Drop all rows that have 1 or 2 sample represented based on year count
    #This happens in the split data into test and train

    X_test, y_test, X_train, y_train, X_val, y_val = split_data_into_test_and_train()
    
    #clean all NaNs and impute values, see the comments int he function inside
    X_test = deal_with_nans_and_imputation(X_test)
    X_train = deal_with_nans_and_imputation(X_train)
    X_val = deal_with_nans_and_imputation(X_val)

    return X_test, y_test, X_train, y_train, X_val, y_val


def build_pipeline_for_clustering():
    df = merge_happiness_alcohol_fertility_migration()
        # Count the occurrences of each Country_Name
    country_counts = df['Country_Name'].value_counts()
    # Filter out rows where the country appears only once or twice
    df = df[df['Country_Name'].isin(country_counts[country_counts > 2].index)]
        # Country Name is  string, we need to label encode that
    # Label Encoding
    df['Country_Encoded'] = df['Country_Name'].astype('category').cat.codes
    # Split the data into train, validation, and test sets
    X_train, X_temp = train_test_split(df, test_size=0.3, random_state=42)  # 70% training, 30% temporary
    X_val, X_test = train_test_split(X_temp, test_size=0.5, random_state=42)  # Split remaining 30% equally between validation and test
    X_train = deal_with_nans_and_imputation(X_train)
    X_val = deal_with_nans_and_imputation(X_val)
    X_test = deal_with_nans_and_imputation(X_test)

    return X_train, X_val, X_test


def deal_with_nans_and_imputation(df):

    #drop columns we do not need,
             # reason'Healthy_Life_Expectancy_At_Birth' -> High Correlation with Other features
             # reason 'Generosity' -> Low correlation with target variable
    df = df.drop(columns = {'Healthy_Life_Expectancy_At_Birth', 'Generosity'})

    # - 'Corruption': Fill out China: 42,  Fill out Turkmenistan: 34
    # The rest gets a mean for the country, if not represented, it gets average of all countries
    #this data is coming from transparency international
    df.loc[df['Country_Name'] == 'China', 'Corruption'] = 0.70
    df.loc[df['Country_Name'] == 'Turkmenistan', 'Corruption'] = 0.84
    df['Corruption'] = df.groupby('Country_Name')['Corruption'].transform(
    lambda x: x.fillna(x.mean()))
    global_mean = df['Corruption'].mean()
    df['Corruption'] = df['Corruption'].fillna(global_mean)
    
    #Alcohol consumption
    # - if no data in the dataset, then we are taking an average of all countries
    # - if one present for the country, use that value nad spread it out
    
    country_means = df.groupby('Country_Name')['Alcohol_consumption'].transform('mean')
    df['Alcohol_consumption'] = df['Alcohol_consumption'].fillna(country_means)
    global_mean = df['Alcohol_consumption'].mean()
    df['Alcohol_consumption'] = df['Alcohol_consumption'].fillna(global_mean)

    #Net Migration
    # - Taiwan Provice of China missing - put a 0 there
    # - The rest: Average for the year
    df.loc[(df['Country_Name'] == 'Taiwan Province of China') 
                & (df['Net_Migration'].isnull()), 'Net_Migration'] = 0
    country_means = df.groupby('Country_Name')['Net_Migration'].transform('mean')
    df['Net_Migration'] = df['Net_Migration'].fillna(country_means)
    global_mean = df['Net_Migration'].mean()
    df['Net_Migration'] = df['Net_Migration'].fillna(global_mean)

    #Regression Imputation
    missing_columns = [ 'Social_Support','Freedom_To_Make_Life_Choices'
                       , 'Positive_Affect','Negative_Affect','Fertility_Rate',
                       'Log_GDP_Per_Capita']
    
    df = deterministic_regression(df, missing_columns)

    #scaling
    scaler = MinMaxScaler()
    numeric_columns = [
        'Log_GDP_Per_Capita', 'Social_Support', 'Freedom_To_Make_Life_Choices',
        'Positive_Affect', 'Negative_Affect', 'Alcohol_consumption', 'Corruption',
        'Fertility_Rate', 'Net_Migration',  'Country_Encoded'
    ]
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

    return df
    


def random_imputation(df, feature):

    number_missing = df[feature].isnull().sum()
    observed_values = df.loc[df[feature].notnull(), feature]
    df.loc[df[feature].isnull(), feature + '_imp'] = np.random.choice(observed_values, number_missing, replace = True)
    
    return df

def deterministic_regression(df, missing_columns):
    #first you run random imputation, which
    #eplaces the missing values with some random observed values of the variable. 
    # The method is repeated for all the variables containing missing values,
    #  after which they serve as parameters in the regression model 
    # to estimate other variable values.

    #Simple Random Imputation is one of the crude methods since it ignores all the other
    #  available data and thus it's very rarely used. But it serves as a good starting 
    # point for regression imputation.

    for feature in missing_columns:
        df[feature + '_imp'] = df[feature]
        df = random_imputation(df, feature)

    # DETERMINISTIC REGRESSION
    # Select only numeric columns for the deterministic regression
    numeric_df = df.select_dtypes(include=['number'])
    
    for feature in missing_columns:
        # Ensure only numeric columns are used, excluding the target feature
        parameters = list(set(numeric_df.columns) - set(missing_columns) - {feature + '_imp'})
        
        # Initialize the Linear Regression model
        model = linear_model.LinearRegression()
        model.fit(X=numeric_df[parameters], y=numeric_df[feature + '_imp'])
        
         # Predict missing values for the feature
        predicted_values = model.predict(numeric_df[parameters])

        # Update the original DataFrame with predicted values where the feature is missing
        df.loc[df[feature].isnull(), feature] = predicted_values[df[feature].isnull()]
    
   # Drop the '_imp' columns if no longer needed
    df = df.drop(columns=[feature + '_imp' for feature in missing_columns])

    return df

X_test, y_test, X_train, y_train, X_val, y_val = build_pipeline()