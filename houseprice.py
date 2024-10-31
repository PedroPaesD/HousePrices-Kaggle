import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.linear_model import Ridge, Lasso

from sklearn.ensemble import GradientBoostingRegressor

import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("train.csv")
testDf = pd.read_csv("test.csv")

####################################################
# Feature Engineering and Data Preprocessing       #
####################################################

# Remove Alley feature, 1369/1460 NaN, remove Fence feature 1179/1460 NaN, remove MiscFeature 1406/1460 NaN, remove MasVnrType 872/1460 NaN,
# remove PoolQC 1453/1460 NaN

df = df.iloc[:, df.columns != 'Alley']
df = df.iloc[:, df.columns != 'Fence']
df = df.iloc[:, df.columns != 'MiscFeature']
df = df.iloc[:, df.columns != 'MasVnrType']
df = df.iloc[:, df.columns != 'PoolQC']

# Substitute NaN in LotFrontage, MasVnrAre, GarageYrBlt by mean

df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].mean())
df['MasVnrArea'] = df['MasVnrArea'].fillna(df['MasVnrArea'].mean())
df['GarageYrBlt'] = df['GarageYrBlt'].fillna(df['GarageYrBlt'].mean())

# Create final basement feature

df['BsmtFinal'] = ''

def basementFeature(row):
    if (row['BsmtCond'] == 'Ex' or row['BsmtCond'] == 'Gd') or (row['BsmtQual'] == 'Ex' or row['BsmtQual'] == 'Gd'):
        return 3
    if row['BsmtCond'] == 'TA' or row['BsmtCond'] == 'Fa' or row['BsmtQual'] == 'TA' or row['BsmtQual'] == 'Fa':
        return 2
    if row['BsmtCond'] == 'Po' or row['BsmtCond'] == 'NA' or row['BsmtQual'] == 'Po' or row['BsmtQual'] == 'NA' :
        return 1
    return 1

df['BsmtFinal'] = df.apply(basementFeature, axis=1)

# Remove rest of bastement features

df = df.iloc[:, df.columns != 'BsmtQual']
df = df.iloc[:, df.columns != 'BsmtCond']
df = df.iloc[:, df.columns != 'BsmtExposure']
df = df.iloc[:, df.columns != 'BsmtFinType1']
df = df.iloc[:, df.columns != 'BsmtFinType2']

# Create final garage feature

df['GrgFinal'] = ''

def garageFeature(row):
    if (row['GarageCond'] == 'Ex' or row['GarageCond'] == 'Gd') or (row['GarageQual'] == 'Ex' or row['GarageQual'] == 'Gd'):
        return 3
    if row['GarageCond'] == 'TA' or row['GarageCond'] == 'Fa' or row['GarageQual'] == 'TA' or row['GarageQual'] == 'Fa':
        return 2
    if row['GarageCond'] == 'Po' or row['GarageCond'] == 'NA' or row['GarageQual'] == 'Po' or row['GarageQual'] == 'NA' :
        return 1
    return 1

df['GrgFinal'] = df.apply(garageFeature, axis=1)

# Remove rest of bastement features

df = df.iloc[:, df.columns != 'GarageQual']
df = df.iloc[:, df.columns != 'GarageCond']
df = df.iloc[:, df.columns != 'GarageType']
df = df.iloc[:, df.columns != 'GarageFinish']

# Fill only value of electrical with Mix

df['Electrical'] = df['Electrical'].fillna('Mix')

# Fill FireplaceQu with TA

df['FireplaceQu'] = df['FireplaceQu'].fillna('TA')

# No more NaN

####################################################
# Repeat for test                                  #
####################################################

# Remove Alley feature, 1369/1460 NaN, remove Fence feature 1179/1460 NaN, remove MiscFeature 1406/1460 NaN, remove MasVnrType 872/1460 NaN,
# remove PoolQC 1453/1460 NaN

testDf = testDf.iloc[:, testDf.columns != 'Alley']
testDf = testDf.iloc[:, testDf.columns != 'Fence']
testDf = testDf.iloc[:, testDf.columns != 'MiscFeature']
testDf = testDf.iloc[:, testDf.columns != 'MasVnrType']
testDf = testDf.iloc[:, testDf.columns != 'PoolQC']

# Substitute NaN in LotFrontage, MasVnrAre, GarageYrBlt by mean

testDf['LotFrontage'] = testDf['LotFrontage'].fillna(testDf['LotFrontage'].mean())
testDf['MasVnrArea'] = testDf['MasVnrArea'].fillna(testDf['MasVnrArea'].mean())
testDf['GarageYrBlt'] = testDf['GarageYrBlt'].fillna(testDf['GarageYrBlt'].mean())

# Create final basement feature

testDf['BsmtFinal'] = ''

def basementFeature(row):
    if (row['BsmtCond'] == 'Ex' or row['BsmtCond'] == 'Gd') or (row['BsmtQual'] == 'Ex' or row['BsmtQual'] == 'Gd'):
        return 3
    if row['BsmtCond'] == 'TA' or row['BsmtCond'] == 'Fa' or row['BsmtQual'] == 'TA' or row['BsmtQual'] == 'Fa':
        return 2
    if row['BsmtCond'] == 'Po' or row['BsmtCond'] == 'NA' or row['BsmtQual'] == 'Po' or row['BsmtQual'] == 'NA' :
        return 1
    return 1

testDf['BsmtFinal'] = testDf.apply(basementFeature, axis=1)

# Remove rest of bastement features

testDf = testDf.iloc[:, testDf.columns != 'BsmtQual']
testDf = testDf.iloc[:, testDf.columns != 'BsmtCond']
testDf = testDf.iloc[:, testDf.columns != 'BsmtExposure']
testDf = testDf.iloc[:, testDf.columns != 'BsmtFinType1']
testDf = testDf.iloc[:, testDf.columns != 'BsmtFinType2']

# Create final garage feature

testDf['GrgFinal'] = ''

def garageFeature(row):
    if (row['GarageCond'] == 'Ex' or row['GarageCond'] == 'Gd') or (row['GarageQual'] == 'Ex' or row['GarageQual'] == 'Gd'):
        return 3
    if row['GarageCond'] == 'TA' or row['GarageCond'] == 'Fa' or row['GarageQual'] == 'TA' or row['GarageQual'] == 'Fa':
        return 2
    if row['GarageCond'] == 'Po' or row['GarageCond'] == 'NA' or row['GarageQual'] == 'Po' or row['GarageQual'] == 'NA' :
        return 1
    return 1

testDf['GrgFinal'] = testDf.apply(garageFeature, axis=1)

# Remove rest of bastement features

testDf = testDf.iloc[:, testDf.columns != 'GarageQual']
testDf = testDf.iloc[:, testDf.columns != 'GarageCond']
testDf = testDf.iloc[:, testDf.columns != 'GarageType']
testDf = testDf.iloc[:, testDf.columns != 'GarageFinish']

# Fill only value of electrical with Mix

testDf['Electrical'] = testDf['Electrical'].fillna('Mix')

# Fill FireplaceQu with TA

testDf['FireplaceQu'] = testDf['FireplaceQu'].fillna('TA')


testDf['TotalBsmtSF'] = testDf['TotalBsmtSF'].fillna(testDf['TotalBsmtSF'].mean())

####################################################
# Build model and predict                          #
####################################################

#print(df.columns)

chosenFeatures = ['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'MasVnrArea', 'TotalBsmtSF', 'MoSold', 'BsmtFinal', 'GrgFinal',
                   'YrSold']

# Normalize

df[chosenFeatures] = df[chosenFeatures]/df[chosenFeatures].max()

X = df[chosenFeatures]
Y = df['SalePrice']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, random_state=16)

model = Ridge(alpha=3.2)
model.fit(X_train, Y_train)

bst = GradientBoostingRegressor(random_state=0)
bst.fit(X_train, Y_train)
bst_pred = bst.predict(X_test)

Y_pred_boost = bst.predict(testDf[chosenFeatures])
Y_pred_ridge = model.predict(testDf[chosenFeatures])

#rmse = root_mean_squared_error(Y_test, Y_pred)
#r_sq = r2_score(Y_test, Y_pred)

#rmse = root_mean_squared_error(Y_test, bst_pred)
#r_sq = r2_score(Y_test, bst_pred)

#print('RMSE:', rmse)
#print('R2:', r_sq)

submission = pd.DataFrame(Y_pred_boost)
submission['Id'] = testDf['Id']
submission.set_index('Id')
submission.rename(columns = {0:'SalePrice'}, inplace = True)
columns_titles = ["Id","SalePrice"]
submission=submission.reindex(columns=columns_titles)

#nan_cols = [i for i in testDf.columns if testDf[i].isnull().any()]
#print(nan_cols)

submission.to_csv('sublinreg.csv', encoding='utf-8', index=False)
