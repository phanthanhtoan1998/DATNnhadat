import warnings

import numpy as np
import pandas as pd
import datetime
import seaborn as sns
# import matplotlib.pyplot as plt
import os


from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

warnings.filterwarnings("ignore")
df = pd.read_csv("housing - housing.csv")
df.head()
print("The initial length of the dataset is", str(len(df)), "rows.")
#Kiểm tra dữ liệu bị missing values
for col in df.columns:
    missing_data = df[col].isna().sum()
    missing_percent = missing_data/len(df) * 100
    print(f"Column {col}:has { missing_percent} %")
df_renamed = df.rename(columns = { "Địa chỉ":"address", "Quận":"district",
                                  "Huyện":"ward", "Loại hình nhà ở":"type_of_housing",
                                 "Giấy tờ pháp lý":"legal_paper", "Số tầng":"num_floors",
                                 "Số phòng ngủ":"num_bed_rooms", "Diện tích":"squared_meter_area",
                                 "Dài":"length_meter", "Rộng":"width_meter", "Giá/m2":"price_in_million_per_square_meter"})
# df_renamed = df_renamed.drop("Unnamed: 0", axis = 1)
df_renamed = df_renamed.dropna()
df_renamed = df_renamed.reset_index()

# The length of the dataset after dropping null values
print("The length of the dataset after dropping null values is", str(len(df_renamed)), "rows.")
# Remove houses with "10 plus" floors and bed rooms, since this cannot be exactly quantified
df_renamed = df_renamed[df_renamed['num_floors'] != 'Nhiều hơn 10']
df_renamed = df_renamed[df_renamed['num_bed_rooms'] != 'nhiều hơn 10 phòng']

# Clean columns and convert numerical columns to float type
df_renamed['district'] = df_renamed['district'].str.replace('Quận ','').str.strip()
df_renamed['ward'] = df_renamed['ward'].str.replace('Phường ','').str.strip()
df_renamed['num_floors'] = df_renamed['num_floors'].astype(float)
df_renamed['num_bed_rooms'] = df_renamed['num_bed_rooms'].astype(float)
df_renamed['squared_meter_area'] = df_renamed['squared_meter_area'].str.replace(' m2','').str.strip().astype(float)
df_renamed['length_meter'] = df_renamed['length_meter'].str.replace('m','') .str.replace(',','.').str.strip().astype(float)
df_renamed['width_meter'] = df_renamed['width_meter'].str.replace('m','').str.replace(' ','') .str.replace(',','.').str.strip().astype(float)
# Clean and convert all prices to million/m2 instead of VND/m2 or billion/m2
df_renamed.loc[df_renamed['price_in_million_per_square_meter'].str.contains('tỷ'), 'price_in_million_per_square_meter'] = df_renamed.loc[df_renamed['price_in_million_per_square_meter'].str.contains('tỷ'), 'price_in_million_per_square_meter'].str.replace(' ','').str.replace(' tỷ ','').str.replace('tỷ','').str.replace(',','.').astype(float) * 1000000000 /df_renamed['squared_meter_area']
df_renamed.loc[df_renamed['price_in_million_per_square_meter'].str.contains('triệu / m2  ', na=False), 'price_in_million_per_square_meter'] = df_renamed.loc[df_renamed['price_in_million_per_square_meter'].str.contains('triệu / m2  ', na=False), 'price_in_million_per_square_meter'].str.replace(' ','').str.replace('triệu/ m2  ','').str.replace(',','.').astype(float) *1000000
df_renamed.loc[df_renamed['price_in_million_per_square_meter'].str.contains(' triệu', na=False), 'price_in_million_per_square_meter'] = df_renamed.loc[df_renamed['price_in_million_per_square_meter'].str.contains(' triệu', na=False), 'price_in_million_per_square_meter'].str.replace(' ','').str.replace('triệu','').str.replace(',','.').astype(float)*1000000 / df_renamed['squared_meter_area']
# df_renamed['price_in_million_per_square_meter'] = df_renamed['price_in_million_per_square_meter'].str.strip().astype(float)
print(' test',df_renamed)
export_csv = df_renamed.to_csv(r'HN.csv', index = None, header=True)

# Create dummies for categorical columns
dummy_type_of_housing = pd.get_dummies(df_renamed.type_of_housing, prefix="housing_type")
dummy_legal_paper = pd.get_dummies(df_renamed.legal_paper, prefix="legal_paper")
dummy_district = pd.get_dummies(df_renamed.district, prefix="district")
dummy_ward = pd.get_dummies(df_renamed.ward, prefix="ward")

df_cleaned = pd.concat([df_renamed, dummy_type_of_housing, dummy_legal_paper, dummy_district, dummy_ward], axis=1)
df_cleaned = df_cleaned.drop(['index', 'address', 'district', 'ward', 'type_of_housing', 'legal_paper'], axis = 1)
df_cleaned.head()
def remove_outlier_IQR(df, series):
    Q1 = df[series].quantile(0.25)
    Q3 = df[series].quantile(0.75)
    IQR = Q3 - Q1
    df_final = df[~((df[series] < (Q1 - 1.5 * IQR)) | (df[series] > (Q3 + 1.5 * IQR)))]
    return df_final


removed_outliers = df_cleaned
columns_to_remove_outliers = ['num_floors', 'num_bed_rooms', 'squared_meter_area', 'length_meter',
                              'width_meter', 'price_in_million_per_square_meter']
for column in columns_to_remove_outliers:
    removed_outliers = remove_outlier_IQR(removed_outliers, column)

print("The final length of the dataset is", str(len(removed_outliers)), "rows.")
housing = removed_outliers

# Separate predictors and response (price) variables
X = housing.loc[:, housing.columns != 'price_in_million_per_square_meter']
y = housing[['price_in_million_per_square_meter']]
to_be_scaled = ['num_floors', 'num_bed_rooms', 'squared_meter_area', 'length_meter', 'width_meter']

# Initiate scaler
PredictorScaler = StandardScaler()
TargetVarScaler = StandardScaler()

X_scaled = X
y_scaled = y

# Storing the fit object for reference and reverse the scaling later
PredictorScalerFit = PredictorScaler.fit(X_scaled[to_be_scaled])
TargetVarScalerFit = TargetVarScaler.fit(y_scaled)

# Generating the standardized values of X and y
X_scaled[to_be_scaled] = PredictorScalerFit.transform(X_scaled[to_be_scaled])
y_scaled = TargetVarScalerFit.transform(y)

X_array = np.array(X_scaled.values).astype("float32")
y_array = np.array(y_scaled).astype("float32")

X_train, X_test, y_train, y_test = train_test_split(X_array, y_array, test_size=0.2, random_state=2032)

# Sanity check to see if all train and test arrays have correct dimensions
if X_train.shape[0] == y_train.shape[0] and X_train.shape[1] == X_test.shape[1] and X_test.shape[0] == y_test.shape[
    0] and y_train.shape[1] == y_test.shape[1]:
    print("All train and test sets have correct dimensions.")
# Turn off TensorFlow messages and warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["KMP_SETTINGS"] = "false"

# Create the base model
def create_regression_ANN(optimizer_trial):
    model = Sequential()
    model.add(Dense(units=10, input_dim=X_train.shape[1], kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mean_squared_error', optimizer=optimizer_trial)
    return model

# Creathe a dictionary for trial parameters
ANN_params = {'batch_size':[10, 20, 30, 50],
             'epochs':[10, 20, 50],
             'optimizer_trial':['adam', 'rmsprop']}

ANN_trial = KerasRegressor(create_regression_ANN, verbose=0)

# Initiate the grid search and storing best parameters for later reference
ANN_grid_search = GridSearchCV(estimator=ANN_trial, param_grid=ANN_params,
                               cv=3, n_jobs = -1).fit(X_train, y_train, verbose=0)
ANN_best_params = ANN_grid_search.best_params_
# Showing the best parameters
ANN_best_params
print("test")
# Fitting the ANN to the Training set
ANN = Sequential()
ANN.add(Dense(units=10, input_dim=X_train.shape[1],
              kernel_initializer='normal', activation='relu'))
ANN.add(Dense(1, kernel_initializer='normal'))
ANN.compile(loss='mean_squared_error', optimizer=ANN_best_params['optimizer_trial'])
ANN.fit(X_train, y_train, batch_size=int(ANN_best_params['batch_size']),
        epochs=int(ANN_best_params['epochs']), verbose=0)

# Generating Predictions on testing data
ANN_predictions = ANN.predict(X_test)

# Scaling the predicted Price data back to original price scale
ANN_predictions = TargetVarScalerFit.inverse_transform(ANN_predictions)

# Scaling the y_test Price data back to original price scale
y_test_orig = TargetVarScalerFit.inverse_transform(y_test)
#
# # Scaling the test data back to original scale
Test_Data = np.concatenate((PredictorScalerFit.inverse_transform(X_test[:, :5]), X_test[:, 5:]), axis=1)

# # Recreating the dataset, now with predicted price using the ANN model
TestingData = pd.DataFrame(data=Test_Data, columns=X.columns)
TestingData['Price'] = y_test_orig
TestingData['ANN_predictions'] = ANN_predictions

TestingData[['Price', 'ANN_predictions']].head()
# # Create a dictionary of random parameters for the model
RF_random_grid = {'n_estimators': [int(x) for x in np.linspace(start = 1, stop = 10, num = 1)],
               'max_features': ['auto', 'sqrt', 'log2'],
               'max_depth': [int(x) for x in np.linspace(10, 100, num = 10)],
               'min_samples_split': [2, 5, 10],
               'min_samples_leaf': [1, 2, 4],
               'bootstrap': [True, False]}
# # Turn off TensorFlow messages and warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["KMP_SETTINGS"] = "false"

# Create the base RF model and fit the random search
RF_regressor = RandomForestRegressor()
RF_random_search = RandomizedSearchCV(estimator=RF_regressor, param_distributions=RF_random_grid, n_iter=5, cv=5,
                                      verbose=0, random_state=2022, n_jobs = -1).fit(X_train, np.ravel(y_train))
RF_best_params = RF_random_search.best_params_
RF_best_params
RF_param_grid = {'n_estimators': [RF_best_params['n_estimators']-100, RF_best_params['n_estimators'], RF_best_params['n_estimators']+100],
               'max_features': ['sqrt', 'log2'],
               'max_depth': [RF_best_params['max_depth'] - 10, RF_best_params['max_depth'], RF_best_params['max_depth']+10],
               'min_samples_split': [5, 10],
               'min_samples_leaf': [1, 2],
               'bootstrap': [True, False]}
# Turn off TensorFlow messages and warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["KMP_SETTINGS"] = "false"

# Create another base RF model and fit the grid search
RF_regressor_2 = RandomForestRegressor()
RF_grid_search = GridSearchCV(estimator=RF_regressor_2, param_grid=RF_param_grid,
                              cv=3, n_jobs=-1, verbose=0).fit(X_train, np.ravel(y_train))

# Showing the best parameters
RF_grid_search.best_params_
# Fitting a RF model with the best parameters
RF = RF_grid_search.best_estimator_

# Generating Predictions on testing data
RF_predictions = RF.predict(X_test)

# Scaling the predicted Price data back to original price scale
RF_predictions = TargetVarScalerFit.inverse_transform(RF_predictions.reshape(-1,1))
print("rf_", RF_predictions)
TestingData['RF_predictions'] = RF_predictions
print("sdád", TestingData['RF_predictions'])
print("testing data", TestingData)

TestingData[['Price', 'ANN_predictions', 'RF_predictions']].head()
# Define a function evaluate the predictions
def Accuracy_Score(orig, pred):
    MAPE = np.mean(100 * (np.abs(orig - pred) / orig))
    return(100-MAPE)
# Showing scores for both the ANN and the RF model
print("Accuracy for the ANN model is:", str(Accuracy_Score(TestingData['Price'], TestingData['ANN_predictions'])))
print("Accuracy for the RF model is:", str(Accuracy_Score(TestingData['Price'], TestingData['RF_predictions'])))