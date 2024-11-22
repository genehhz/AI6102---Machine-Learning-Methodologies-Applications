# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 01:26:25 2024

@author: geneh
"""

import os
import numpy as np
import pandas as pd
import re
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import json


os.chdir('C:/Users/geneh/Desktop/ml')

pd.set_option('display.max_rows', None)

df = pd.read_csv("train.csv")
dt = pd.read_csv("test.csv")

# Check NULL values for dt and df:
print(df.isnull().sum())
print(dt.isnull().sum())

#clean_title (pretty self explanatory to fill the missing rows with 'no')
df['clean_title'].fillna('no', inplace=True)
dt['clean_title'].fillna('no', inplace=True)

# Missing values for accident:
df['accident'].fillna('None reported', inplace=True)
dt['accident'].fillna('None reported', inplace=True)
# accident:
df['accident']=df['accident'].map({'None reported': 0,'At least 1 accident or damage reported': 1})
dt['accident']=dt['accident'].map({'None reported': 0,'At least 1 accident or damage reported': 1})

# All NULL values for accident and clean_title handled:
print(dt.isnull().sum())
print(dt.isnull().sum())

# Standardizing %-Speed format:
df.transmission = df.transmission.replace({"Single-Speed Fixed Gear":"1-Speed Fixed Gear"})
dt.transmission = dt.transmission.replace({"Single-Speed Fixed Gear":"1-Speed Fixed Gear"})

obj_default_na = 'Missing'

def extract_gear_and_txtype(transmission_info):
    pattern = re.search(r'(\d{1,2}[\s-]?speed?)?\s*(Automatic|Electronically Controlled Automatic|At\/Mt|A\/T|AT|M\/T|CVT|Manual|Variable|Transmission Overdrive|Fixed|DCT|Mt|Transmission w/Dual Shift Mode)?\s*',transmission_info,re.IGNORECASE)
    
    gear = pattern.group(1) if pattern.group(1) else None
    txtype = pattern.group(2) if pattern.group(2) else "Other"
    return gear, txtype

def load_gear_and_txtype(df):
    gear = []
    transmission_type = []

    for tx in df.transmission:
        ngear,txtype = extract_gear_and_txtype(tx)

        if ngear!=None:
            ngear = ngear.split("-")[0].split(" ")[0] #to tackle both 6-speed & 6 speed
            if ngear.lower()=="single":ngear=1
            else:ngear = int(ngear)

        if txtype!=None: # 
            if txtype=="At/Mt": txtype="AMT"
            elif txtype.lower() in ['a/t','at','transmission overdrive']: txtype = "Automatic"
            elif txtype.lower() in ['m/t','mt']: txtype = "Manual"
            elif txtype.lower()=="variable": txtype="CVT"
            elif txtype=="Transmission w/Dual Shift Mode": txtype="Dual_Shift"
            elif txtype=="Electronically Controlled Automatic": txtype="Electronically_controlled"

        gear.append(ngear)
        transmission_type.append(txtype)
    
    def get_special_features(trans):
        features = []
        if 'Dual Shift Mode' in trans:
            features.append('Dual Shift Mode')
        if 'Auto-Shift' in trans:
            features.append('Auto-Shift')
        if 'Overdrive' in trans:
            features.append('Overdrive')
        if 'Electronically Controlled' in trans:
            features.append('Electronically Controlled')
        if 'Variable' in trans:
            features.append('Variable')
        return ', '.join(features) if features else obj_default_na
    
    
    def get_transmission_designation(trans):
        if 'A/T' in trans:
            return 'A/T'
        elif 'M/T' in trans:
            return 'M/T'
        elif 'CVT' in trans:
            return 'CVT'
        elif 'DCT' in trans:
            return 'DCT'
        else:
            return obj_default_na
            
        
    df["gears"] = gear # number of gears
    df["transmission_type"] = transmission_type
    df['special_features'] = df['transmission'].apply(get_special_features)
    df['transmission_designation'] = df['transmission'].apply(get_transmission_designation)

    
    return df
    
  
df = load_gear_and_txtype(df)
dt = load_gear_and_txtype(dt)
  
# There are a lot of missing values for 'gears':
print(df.isnull().sum())
print(dt.isnull().sum())
  
    
# Find the proportions of 'misisng' in each extracted column:
missing_prop_per_column = ((df == 'Missing').sum()/len(df))*100
print("proportion of 'missing' values per column:\n", missing_prop_per_column)
    
  
df=df.drop(['transmission_designation'],axis=1)
dt=dt.drop(['transmission_designation'],axis=1)
  
#'fuel_type' 
## changed '-' and 'not supported' to Nan values
print(df.fuel_type.unique())
print(dt.fuel_type.unique())
df['fuel_type'] = df['fuel_type'].replace('–', np.nan)
df['fuel_type'] = df['fuel_type'].replace('not supported', np.nan)
dt['fuel_type'] = dt['fuel_type'].replace('–', np.nan)
dt['fuel_type'] = dt['fuel_type'].replace('not supported', np.nan)
  
    
print(df.isnull().sum())
  
#list of electric only car brands
Evs_brand = ['Tesla', 'Rivian', 'Lucid','Polestar']
#if any of the car brands is in the Evs_brand, change their fuel type to 'Electric'
df.loc[df['brand'].isin(Evs_brand), 'fuel_type'] = 'Electric'
dt.loc[dt['brand'].isin(Evs_brand), 'fuel_type'] = 'Electric'
  
    
def extract_engine(s: str):
    # Extract fuel type
    fuel_types = ['Gasoline', 'Diesel', 'Electric', 'Hybrid', 'Flex Fuel']
    fuel_ext = next((fuel for fuel in fuel_types if fuel in s), np.nan)  # Return the first matching fuel type, or an empty string

    # Extract horsepower
    hpgroup = re.search(r'(\d+(\.\d+)?)\s*hp', s, re.IGNORECASE)
    engine_hp = float(hpgroup.group(1)) if hpgroup else np.nan #impute with NA
    
    # Extract liters (L or Liter)
    litergroup = re.search(r'(\d+(\.\d+)?)\s*[Ll](?:iter)?', s, re.IGNORECASE)
    engine_liter = float(litergroup.group(1)) if litergroup else np.nan
    
    # Extract cylinder count
    cylindergroup = re.search(r'(\d+)\s*cylinders?|([ivxlcdm]+)\s*cylinders?', s, re.IGNORECASE)
    if cylindergroup:
        engine_cyl = int(cylindergroup.group(1)) if cylindergroup.group(1) else cylindergroup.group(2).upper()
    else:
        engine_cyl = np.nan #impute integer with NAs
    
    # extract
    
    return engine_hp, engine_liter, engine_cyl, fuel_ext
 

df[['engine_hp','engine_liter','engine_cyl','engine_fuel']]=df['engine'].apply(extract_engine).apply(pd.Series)
dt[['engine_hp','engine_liter','engine_cyl','engine_fuel']]=dt['engine'].apply(extract_engine).apply(pd.Series)


#using above extracted columns to get actual fuel types
#under engine, if it has "L" and 'electric" , this categorise into hybrid
condition = (pd.notna(df['engine_liter'])) & (df['engine_fuel'] == 'Electric')
df.loc[condition, 'fuel_type'] = "Hybrid"
condition2 = (pd.notna(dt['engine_liter'])) & (dt['engine_fuel'] == 'Electric')
dt.loc[condition2, 'fuel_type'] = "Hybrid"

# Replace Flex Fuel:
df['fuel_type'] = df['fuel_type'].replace('E85 Flex Fuel', 'Flex Fuel')
dt['fuel_type'] = dt['fuel_type'].replace('E85 Flex Fuel', 'Flex Fuel')

# replace the missing values in fuel_type column with the extracted fuel from engine 'engine_fuel'
df['fuel_type'] = df['fuel_type'].fillna(df['engine_fuel'])
dt['fuel_type'] = dt['fuel_type'].fillna(dt['engine_fuel'])

#impute null values of a column (target_cols) based on the most frequent value in the same brand and model group. 
# If the combination of brand and model doesn't have a mode, it falls back to using the mode of the brand alone.

def fill_na_with_same_brand_model_mode(df, grouping_cols, target_cols):
    
    # Mode Calculator
    def calculate_mode(series):
        mode = series.mode()
        return mode.iloc[0] if not mode.empty else np.nan
    
    # NA filler function
    def impute_na(row):
        if pd.notna(row[col]):
            return row[col]
        if pd.notna(row['brand_model_mode']):
            return row['brand_model_mode']
        if pd.notna(row['brand_mode']):
            return row['brand_mode']

        return row[col]
    
    #-------------
    for col in target_cols:
        # find mode of same brand & model
        brand_model_mode = df.groupby(grouping_cols)[col].apply(calculate_mode).rename("brand_model_mode")
        
        # find mode of same brand
        brand_mode = df.groupby(grouping_cols[0])[col].apply(calculate_mode).rename("brand_mode")
        
        #merging the series to have corresponding mode for each brand & band_model combo in each rows
        df = df.merge(brand_model_mode, on=grouping_cols, how='left')
        df = df.merge(brand_mode, on=grouping_cols[0], how='left')
        
        #applying na_filler function
        df[col] = df.apply(impute_na, axis=1)
         #deleting newly added cols
        df.drop(columns=['brand_model_mode','brand_mode'],inplace=True)
    
    return df

df = fill_na_with_same_brand_model_mode(df, ['brand','model'], ['gears','engine_hp','engine_liter','engine_cyl', 'fuel_type'])
dt= fill_na_with_same_brand_model_mode(dt, ['brand','model'], ['gears','engine_hp','engine_liter','engine_cyl', 'fuel_type'])

df['engine_hp'].fillna(1001, inplace=True)
df['engine_cyl'].fillna(16, inplace=True)
for i, row in dt.iterrows():
    if pd.isna(row['engine_cyl']):  # Check if engine_cyl is NaN
        if row['brand'] == 'Bugatti':
            dt.at[i, 'engine_cyl'] = 16  # Set to 16 for Bugatti
            dt.at[i, 'engine_hp'] = 1001 #set 1001
        elif row['brand'] == 'Karma':
            dt.at[i, 'engine_cyl'] = 4   # Set to 4 for Karma

print(df.isnull().sum())
print(dt.isnull().sum())

# Function to find the first matching base color
def find_first_matching_color(colors:str):
    colors = colors.lower()
    base_colors = ['white', 'black', 'grey', 'gray', 'blue', 'red', 'yellow', 'silver', 'green', 'beige', 'gold', 'orange', 'brown', 'ebony', 'purple']
    color = ''  # Default value
    for i in base_colors:
        if i in colors:
            color = i
            return color
    return "uncommon"  # Return None if no matching color is found

# apply to both ext_col and int_col
df['ext_col']=df['ext_col'].apply(find_first_matching_color)
df['int_col']=df['int_col'].apply(find_first_matching_color)

df.int_col.value_counts()
   
dt['ext_col']=dt['ext_col'].apply(find_first_matching_color)
dt['int_col']=dt['int_col'].apply(find_first_matching_color)

# Distributions of colours:
print(df.ext_col.value_counts(normalize=True)*100)
print(dt.int_col.value_counts(normalize=True)*100)

# spelling:
df['ext_col'] = df['ext_col'].replace({"grey":"gray"})
df['int_col'] = df['int_col'].replace({"grey":"gray"})
dt['ext_col'] = dt['ext_col'].replace({"grey":"gray"})
dt['int_col'] = dt['int_col'].replace({"grey":"gray"})

# Creation of additional featurs:
def feature(df):
    df['brand'] = df['brand'].str.lower()
    current_year = 2024
    df['Vehicle_Age'] = current_year - df['model_year']
    df['Mileage_per_Year'] = df['milage'] / (df['Vehicle_Age'] + 10e-5)
    luxury_brands = ["mercedes-benz","bmw","audi","porsche","lexus","cadillac","jaguar",
                     "bentley","genesis","maserati","lamborghini","rolls-royce","ferrari",
                     "mclaren","aston","lotus","bugatti","maybach"] #based on?
    df['Is_Luxury_Brand'] = df['brand'].apply(lambda x: 1 if x in luxury_brands else 0)
    
    return df

df = feature(df)
dt = feature(dt)


df=df.drop(['id','engine','transmission','model_year','model','engine_fuel'],axis=1)
dt=dt.drop(['id','engine','transmission','model_year','model','engine_fuel'],axis=1)


print(df.isnull().sum())
print(dt.isnull().sum())

df.info()

categorical_columns = df.select_dtypes(include=['object']).columns
print(categorical_columns)

numerical_columns = df.select_dtypes(include=['int64','float64']).columns
print(numerical_columns)

# Number of unqiue categories per categorical variable:
unique_values = {col: df[col].nunique() for col in categorical_columns}

for col, unique_count in unique_values.items():
    print(f"{col}: {unique_count} unique values")

print('this would result in the creation of', sum(unique_values.values()), 'new columns in the dataset' )


y=df['price']

X = df.drop(['price'], axis=1)
Xt = dt

print(X.shape,y.shape)
print(Xt.shape)

cat_feats = ['brand', 'fuel_type',  'ext_col',
       'int_col', 'clean_title', 'transmission_type',
       'special_features']

cat_feats_dt = ['brand','fuel_type','ext_col','int_col','clean_title','transmission_type','special_features']

numeric_feats = ['milage', 'accident', 'price', 'gears','engine_hp', 'engine_liter',
       'engine_cyl','Vehicle_Age', 'Mileage_per_Year', 'Is_Luxury_Brand']


# Initialize ColumnTransformer with OneHotEncoder for categorical columns
preprocessor_df = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), cat_feats)
    ],
    remainder='passthrough'  # Keep the other (numerical) columns as they are
)

preprocessor_dt = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), cat_feats_dt)
    ],
    remainder='passthrough'  # Keep the other (numerical) columns as they are
)

# Fit and transform the data
encoded_data_df = preprocessor_df.fit_transform(X)
encoded_data_dt = preprocessor_dt.fit_transform(Xt)

# Get feature names after transformation
encoded_feature_names_df = preprocessor_df.get_feature_names_out()
encoded_feature_names_dt = preprocessor_dt.get_feature_names_out()


# Convert to DataFrame
encoded_df = pd.DataFrame(encoded_data_df, columns=encoded_feature_names_df)
encoded_dt = pd.DataFrame(encoded_data_dt, columns=encoded_feature_names_dt)

encoded_df.to_csv('encoded_df.csv')

# Assess and remove redundant columns 

encoded_df_columns = set(encoded_df.columns)
encoded_dt_columns = set(encoded_dt.columns)

# Find the columns present in encoded_df but not in encoded_dt
extra_columns_in_df = encoded_df_columns - encoded_dt_columns

# Find the columns present in encoded_dt but not in encoded_df
extra_columns_in_dt = encoded_dt_columns - encoded_df_columns

# Print the results
print("Columns present in encoded_df but not in encoded_dt:")
print(extra_columns_in_df)

print("Columns present in encoded_dt but not in encoded_df:")
print(extra_columns_in_dt)

columns_to_remove = ['cat__brand_polestar', 'cat__brand_smart']
encoded_df = encoded_df.drop(columns=columns_to_remove)

encoded_df = pd.concat([encoded_df, y], axis=1)

encoded_df.to_csv('encoded_df.csv')
encoded_dt.to_csv('encoded_dt.csv')

# Assuming your data is in a pandas DataFrame called 'encoded_df'
data = encoded_df.values

# Split the data into features (X) and target (y)
X = data[:, :-1]  # All columns except the last one
y = data[:, -1]   # Last column (price)

# Convert the data to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# Define the neural network model
class Net(nn.Module):
    def __init__(self, input_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        return x

def RMSELoss(yhat, y):
    return torch.sqrt(torch.mean((yhat - y) ** 2))

# Global variables to store losses as dictionaries
global_fold_train_losses = {}  # Will store losses for each fold as {fold_num: losses}
global_fold_val_losses = {}    # Will store losses for each fold as {fold_num: losses}

def calculate_average_losses():
    """Calculate average losses from fold losses dictionaries"""
    if not global_fold_train_losses or not global_fold_val_losses:
        return [], []
        
    num_epochs = len(list(global_fold_train_losses.values())[0])  # Get length from first fold
    avg_train_losses = []
    avg_val_losses = []
    
    for epoch in range(num_epochs):
        # Get all fold losses for this epoch
        train_losses_at_epoch = [losses[epoch] for losses in global_fold_train_losses.values()]
        val_losses_at_epoch = [losses[epoch] for losses in global_fold_val_losses.values()]
        
        # Calculate averages
        avg_train_losses.append(float(np.mean(train_losses_at_epoch)))
        avg_val_losses.append(float(np.mean(val_losses_at_epoch)))
    
    return avg_train_losses, avg_val_losses

def cross_validate_model(model, X_train, y_train, params, n_splits=5, num_epochs=1000):
    # Initialize variables
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Clear previous fold losses
    global_fold_train_losses.clear()
    global_fold_val_losses.clear()
    
    # Cross-validation loop
    for fold, (train_ind, valid_ind) in enumerate(cv.split(X_train)):
        print(f"Starting fold {fold+1}/{n_splits}")
        # Data splitting
        X_fold_train = X_train[train_ind]
        y_fold_train = y_train[train_ind]
        X_val = X_train[valid_ind]
        y_val = y_train[valid_ind]
        
        # Model initialization and training
        model = Net(**params)
        criterion = RMSELoss
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        train_losses = []
        val_losses = []
        
        for epoch in range(num_epochs):
            # Forward pass
            outputs = model(X_fold_train)
            train_loss = criterion(outputs, y_fold_train.unsqueeze(1))
            
            # Backward and optimize
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            
            # Evaluate on validation set
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val.unsqueeze(1))
            train_losses.append(float(train_loss.item()))
            val_losses.append(float(val_loss.item()))
            
            if (epoch+1) % 50 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Train RMSE: {train_loss.item():.4f}, Val RMSE: {val_loss.item():.4f}")
        
        # Store individual fold losses in dictionaries
        global_fold_train_losses[fold] = train_losses
        global_fold_val_losses[fold] = val_losses
        print(f"Fold {fold+1} completed.")
    
    return model

def plot_loss_curves(save_path='loss_curve.png', plot_folds=False):
    """
    Plot the training and validation loss curves using the global variables
    
    Parameters:
    save_path (str): Path where the plot should be saved
    plot_folds (bool): If True, plot individual fold losses along with averages
    """
    if not global_fold_train_losses or not global_fold_val_losses:
        raise ValueError("No loss data available. Run cross_validate_model first.")
    
    # Calculate average losses
    avg_train_losses, avg_val_losses = calculate_average_losses()
    
    plt.figure(figsize=(12, 6))
    
    if plot_folds:
        # Plot individual fold losses with light alpha
        for fold in global_fold_train_losses.keys():
            plt.plot(global_fold_train_losses[fold], alpha=0.2, color='blue', 
                    label=f'Fold {fold} Train' if fold == 0 else "")
            plt.plot(global_fold_val_losses[fold], alpha=0.2, color='orange',
                    label=f'Fold {fold} Val' if fold == 0 else "")
    
    # Plot average losses
    plt.plot(avg_train_losses, label='Training loss', linewidth=2)
    plt.plot(avg_val_losses, label='Validation loss', linewidth=2)
    
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.title('Training and Validation Loss Curve')
    plt.legend()
    plt.show()
    plt.savefig(save_path)
    plt.close()

# Usage example:
model = cross_validate_model(Net, X_tensor, y_tensor, 
                           params={'input_size': X_tensor.shape[1]}, 
                           num_epochs=1000)

# Get average losses if needed
avg_train_losses, avg_val_losses = calculate_average_losses()

plot_loss_curves()

# Save the trained model
torch.save(model.state_dict(), 'price_prediction_model.pth')

test_data = encoded_dt.values

X_test_tensor = torch.tensor(test_data, dtype=torch.float32)

model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    test_predictions = test_outputs.squeeze().detach().numpy()

# Create a DataFrame with the predictions
test_df = pd.DataFrame({'id': range(188533, 188533 + len(test_predictions)), 'price': test_predictions})

# Save the predictions to a CSV file
test_df.to_csv('test_predictions_181124.csv', index=False)

kfold_last_val = [global_fold_val_losses[fold][-1] for fold in range(5)]
mean_val = np.average(kfold_last_val)
kfold_last_train = [global_fold_train_losses[fold][-1] for fold in range(5)]
mean_train = np.average(kfold_last_train)


# output training and val loss as json

combined_losses = {
    "train_losses": global_fold_train_losses,
    "val_losses": global_fold_val_losses
}

# Save to file
with open('losses.json', 'w') as f:
    json.dump(combined_losses, f, indent=2)