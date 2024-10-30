import pandas as pd

df = pd.read_csv("train.csv")

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

# Feature Engineering

print(df)


