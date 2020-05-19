import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer

# Read the data
X_full = pd.read_csv('./input/train.csv', index_col='Id')
X_test_full = pd.read_csv('./input/test.csv', index_col='Id')

# Obtain target and predictors
y = X_full.SalePrice
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = X_full[features].copy()
X_test = X_test_full[features].copy()

# Break off validation set from training data
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

# Define the models
model_1 = RandomForestRegressor(n_estimators=50, random_state=0)
model_2 = RandomForestRegressor(n_estimators=100, random_state=0)
model_3 = RandomForestRegressor(n_estimators=100, criterion='mae', random_state=0)
model_4 = RandomForestRegressor(n_estimators=200, min_samples_split=20, random_state=0)
model_5 = RandomForestRegressor(n_estimators=100, max_depth=7, random_state=0)

models = [model_1, model_2, model_3, model_4, model_5]

# Function for comparing different models
def score_model(model, X_t=X_train, X_v=X_valid, y_t=y_train, y_v=y_valid):
    model.fit(X_t, y_t)
    preds = model.predict(X_v)
    return mean_absolute_error(y_v, preds)

for i in range(0, len(models)):
    mae = score_model(models[i])
    print("Model %d MAE: %d" % (i+1, mae))

#MISSING VALUES

# Load the data
data = pd.read_csv('./input/melb_data.csv')

# Select target
y = data.Price

# To keep things simple, we'll use only numerical predictors
# An object data type in Pandas represents Strings
#axis=1 drops columns, default = 0 (rows)
melb_predictors = data.drop(['Price'], axis=1)
X = melb_predictors.select_dtypes(exclude=['object'])

# Divide data into training and validation subsets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                      random_state=0)

# Function for comparing different approaches
def score_dataset(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=10, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)

# Get names of columns with missing values
# isnull() returns true where data is missing
# any() is a function of numpy array
cols_with_missing = [col for col in X_train.columns
                     if X_train[col].isnull().any()]
print("****************************************************")
print("cols_with_missing: \n", cols_with_missing)
print("****************************************************")

# Drop columns in training and validation data
reduced_X_train = X_train.drop(cols_with_missing, axis=1)
reduced_X_valid = X_valid.drop(cols_with_missing, axis=1)

print("MAE from Approach 1 (Drop columns with missing values):")
print(score_dataset(reduced_X_train, reduced_X_valid, y_train, y_valid))

# Imputation with SimpleImputer
my_imputer = SimpleImputer()
imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))
imputed_X_valid = pd.DataFrame(my_imputer.transform(X_valid))

# Imputation removed column names; put them back
imputed_X_train.columns = X_train.columns
imputed_X_valid.columns = X_valid.columns

print("MAE from Approach 2 (Imputation):")
print(score_dataset(imputed_X_train, imputed_X_valid, y_train, y_valid))

# Make copy to avoid changing original data (when imputing)
X_train_plus = X_train.copy()
X_valid_plus = X_valid.copy()

print(X_train.head())
print("X_train type: ", type(X_train))

# Make new columns indicating what will be imputed
# DataFrame_Name['column_name'] on its own will retrieve column values, but to filter the column values we can spcify a condition on the right hand side.
# Syntax DataFrame_Name['column_name'] creates a column and fills it based on the condition on the right hand side
# See https://datatofish.com/select-rows-pandas-dataframe/ for details
# DataFrame[column] will create a column if the column name is not found and in this case it'll usually be on the left hand side of the "=" sign...
# But, the same syntax is used to retreive a current column name and in this case it'll be on the right hand side of "="
# SEE: https://www.geeksforgeeks.org/dealing-with-rows-and-columns-in-pandas-dataframe/ AND https://mode.com/python-tutorial/pandas-dataframe/
for col in cols_with_missing:
    X_train_plus[col + '_was_missing'] = X_train_plus[col].isnull() # Remember that we had dropped the columns for having "any" value null
    X_valid_plus[col + '_was_missing'] = X_valid_plus[col].isnull()

print(X_train_plus.head())
print("X_train_plus type: ", type(X_train_plus))

# Imputation
my_imputer = SimpleImputer()
imputed_X_train_plus = pd.DataFrame(my_imputer.fit_transform(X_train_plus))
imputed_X_valid_plus = pd.DataFrame(my_imputer.transform(X_valid_plus))

# Imputation removed column names; put them back
imputed_X_train_plus.columns = X_train_plus.columns
imputed_X_valid_plus.columns = X_valid_plus.columns

print("MAE from Approach 3 (An Extension to Imputation):")
print(score_dataset(imputed_X_train_plus, imputed_X_valid_plus, y_train, y_valid))

# Shape of training data (num_rows, num_columns)
print("****************************************************")
print("Rows, Columns ", X_train.shape)
print("X_train type: ", type(X_train))

# Number of missing values in each column of training data
# https://stackoverflow.com/questions/41681693/pandas-isnull-sum-with-column-headers
# https://dzone.com/articles/pandas-find-rows-where-columnfield-is-null
# isnull() returns true where data is missing and indicates TRUE and FALSE under each column
# Remember that isnull() still returns a DataFrame
# sum() converts the data type to Series and returns sum for each series
# https://www.geeksforgeeks.org/python-pandas-series-sum/
# https://datascience.stackexchange.com/questions/12645/how-to-count-the-number-of-missing-values-in-each-row-in-pandas-dataframe
# Calling sum() of the DataFrame returned by isnull() will give a series containing data about count of NaN in each column
# https://thispointer.com/python-pandas-count-number-of-nan-or-missing-values-in-dataframe-also-row-column-wise/
missing_val_count_by_column = (X_train.isnull().sum())
print("****************************************************")
print("Data type of missing_val_count_by_column: \n", type(missing_val_count_by_column))
print("****************************************************")
print("missing_val_count_by_column: \n", missing_val_count_by_column)
print("****************************************************")
print("missing_val_count_by_column > 0: \n", missing_val_count_by_column[missing_val_count_by_column > 0])

