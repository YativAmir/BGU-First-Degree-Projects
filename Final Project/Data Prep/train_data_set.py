# libraries
import pandas as pd
 
# load PKL file 
df = pd.read_csv("C:\\Users\\doron\\OneDrive\\שולחן העבודה\\פרויקט גמר\\train wothout unneeded columns.csv")

# create data frame without None values ​​in 'Id'
df = df.dropna(subset=['Id'])


# make 'Id' as index
def set_id_as_index(df):

    if 'Id' not in df.columns:
        raise ValueError("DataFrame must contain 'ID' column.")
    
    return df.set_index('Id')

df['Id'] = df['Id'].astype(int)
df = set_id_as_index(df)


# create category columns with 1 or 0 
def create_category_columns(df, column_name):
    categories = df[column_name].unique()
    
    for category in categories:
        new_column_name = f"{column_name}_{category}"
        df[new_column_name] = (df[column_name] == category).astype(int)
    df = df.drop(columns=column_name)
    
    return df

# Example usage
# df = create_category_columns(df, 'MSSubClass')


# for normalized percentile column
def add_percentile_column(df, column):

    percentiles = df[column].rank(pct=True, method='average')
    percentile_column_name = f"{column}_percentile"   
    df[percentile_column_name] = percentiles
    df = df.drop(columns=column)    
    return df

# Example usage
# df = add_percentile_column(df, 'LotFrontage')


# for normalized rating columns
def rating_column(df, column):

    df[column] = df[column] / 10   
    return df

# Example usage
# df = rating_column(df, 'OverallQual')


# for Categorical with a grade values
def update_column_with_numerical_values(df, column, My_List):
    updated_df = df.copy()   
    if column not in updated_df.columns:
        print(f"Column '{column}' not found in DataFrame.")
        return updated_df    
    updated_df[column] = [My_List.index(value) + 1 for value in updated_df[column]]      
    return updated_df

# Example usage
#My_List = ['20',
#          '30']
#df = update_column_with_numerical_values(df, 'MSSubClass' , My_List)
#df = add_percentile_column(df, 'MSSubClass')


# ----------------- Dataset's conversion ---------------------------

# Category with order MSSubClass
# MSSubClass
df['MSSubClass'] = df['MSSubClass'].apply(str)
My_List = ['20.0',
           '30.0', 
           '40.0', 
           '45.0',
           '50.0', 
           '60.0',
           '70.0',
           '75.0', 
           '80.0',
           '85.0',
           '90.0', 
           '120.0',
           '150.0',
           '160.0', 
           '180.0',
           '190.0']
df = update_column_with_numerical_values(df, 'MSSubClass' , My_List)
df['MSSubClass'] = df['MSSubClass'] / len(My_List)

#OverallQual
df = rating_column(df, 'OverallQual')

#OverallCond
df = rating_column(df, 'OverallCond')

#ExterQual
My_List = ['Po',
           'Fa', 
           'TA', 
           'Gd',
           'Ex']
df = update_column_with_numerical_values(df, 'ExterQual' , My_List)
df['ExterQual'] = df['ExterQual'] / len(My_List)

#ExterCond
My_List = ['Po',
           'Fa', 
           'TA', 
           'Gd',
           'Ex']
df = update_column_with_numerical_values(df, 'ExterCond' , My_List)
df['ExterCond'] = df['ExterCond'] / len(My_List)

#BsmtQual
df['BsmtQual'] = df['BsmtQual'].fillna("NA")
My_List = ['NA',
           'Po',
           'Fa', 
           'TA', 
           'Gd',
           'Ex']
df = update_column_with_numerical_values(df, 'BsmtQual' , My_List)
df['BsmtQual'] = df['BsmtQual'] / len(My_List)

#BsmtCond
df['BsmtCond'] = df['BsmtCond'].fillna("NA")
My_List = ['NA',
           'Po',
           'Fa', 
           'TA', 
           'Gd',
           'Ex']
df = update_column_with_numerical_values(df, 'BsmtCond' , My_List)
df['BsmtCond'] = df['BsmtCond'] / len(My_List)

#BsmtExposure
df['BsmtExposure'] = df['BsmtExposure'].fillna("NA")
My_List = ['NA',
           'No',
           'Mn', 
           'Av', 
           'Gd',]
df = update_column_with_numerical_values(df, 'BsmtExposure' , My_List)
df['BsmtExposure'] = df['BsmtExposure'] / len(My_List)

#BsmtFinType1
df['BsmtFinType1'] = df['BsmtFinType1'].fillna("NA")
My_List = ['NA',
           'Unf',
           'LwQ', 
           'Rec', 
           'BLQ',
           'ALQ',
           'GLQ']
df = update_column_with_numerical_values(df, 'BsmtFinType1' , My_List)
df['BsmtFinType1'] = df['BsmtFinType1'] / len(My_List)

#BsmtFinType2
df['BsmtFinType2'] = df['BsmtFinType2'].fillna("NA")
My_List = ['NA',
           'Unf',
           'LwQ', 
           'Rec', 
           'BLQ',
           'ALQ',
           'GLQ']
df = update_column_with_numerical_values(df, 'BsmtFinType2' , My_List)
df['BsmtFinType2'] = df['BsmtFinType2'] / len(My_List)

#HeatingQC
My_List = ['Po',
           'Fa', 
           'TA', 
           'Gd',
           'Ex']
df = update_column_with_numerical_values(df, 'HeatingQC' , My_List)
df['HeatingQC'] = df['HeatingQC'] / len(My_List)

#KitchenQual
My_List = ['Po',
           'Fa', 
           'TA', 
           'Gd',
           'Ex']
df = update_column_with_numerical_values(df, 'KitchenQual' , My_List)
df['KitchenQual'] = df['KitchenQual'] / len(My_List)

#FireplaceQu
df['FireplaceQu'] = df['FireplaceQu'].fillna("NA")
My_List = ['NA',
           'Po',
           'Fa', 
           'TA', 
           'Gd',
           'Ex']
df = update_column_with_numerical_values(df, 'FireplaceQu' , My_List)
df['FireplaceQu'] = df['FireplaceQu'] / len(My_List)

#GarageFinish
df['GarageFinish'] = df['GarageFinish'].fillna("NA")
My_List = ['NA',
           'Unf',
           'RFn', 
           'Fin']
df = update_column_with_numerical_values(df, 'GarageFinish' , My_List)
df['GarageFinish'] = df['GarageFinish'] / len(My_List)

#GarageQual
df['GarageQual'] = df['GarageQual'].fillna("NA")
My_List = ['NA',
           'Po',
           'Fa', 
           'TA', 
           'Gd',
           'Ex']
df = update_column_with_numerical_values(df, 'GarageQual' , My_List)
df['GarageQual'] = df['GarageQual'] / len(My_List)

#GarageCond
df['GarageCond'] = df['GarageCond'].fillna("NA")
My_List = ['NA',
           'Po',
           'Fa', 
           'TA', 
           'Gd',
           'Ex']
df = update_column_with_numerical_values(df, 'GarageCond' , My_List)
df['GarageCond'] = df['GarageCond'] / len(My_List)

#PoolQC
df['PoolQC'] = df['PoolQC'].fillna("NA")
My_List = ['NA',
           'Fa', 
           'TA', 
           'Gd',
           'Ex']
df = update_column_with_numerical_values(df, 'PoolQC' , My_List)
df['PoolQC'] = df['PoolQC'] / len(My_List)


#Categories withou order

df = create_category_columns(df, 'MSZoning')
df = create_category_columns(df, 'Alley')
df = create_category_columns(df, 'LotShape')
df = create_category_columns(df, 'LandContour')
df = create_category_columns(df, 'LotConfig')
df = create_category_columns(df, 'LandSlope')
df = create_category_columns(df, 'Neighborhood')
df = create_category_columns(df, 'Condition1')
df = create_category_columns(df, 'Condition2')
df = create_category_columns(df, 'BldgType')
df = create_category_columns(df, 'HouseStyle')
df = create_category_columns(df, 'RoofStyle')
df = create_category_columns(df, 'Exterior1st')
df = create_category_columns(df, 'Exterior2nd')
df = create_category_columns(df, 'MasVnrType')
df = create_category_columns(df, 'Foundation')
df = create_category_columns(df, 'Heating')
df = create_category_columns(df, 'Electrical')
df = create_category_columns(df, 'Functional')
df = create_category_columns(df, 'GarageType')
df = create_category_columns(df, 'PavedDrive')
df = create_category_columns(df, 'Fence')
df = create_category_columns(df, 'SaleType')
df = create_category_columns(df, 'SaleCondition')

# Binary category

df['CentralAir'] = df['CentralAir'].apply(lambda x: 1 if x == 'Y' else 0)

# Numerical features
df = add_percentile_column(df, 'LotFrontage')
df = add_percentile_column(df, 'LotArea')
df = add_percentile_column(df, 'YearBuilt')
df = add_percentile_column(df, 'YearRemodAdd')
df = add_percentile_column(df, 'MasVnrArea')
df = add_percentile_column(df, 'BsmtFinSF1')
df = add_percentile_column(df, 'BsmtFinSF2')
df = add_percentile_column(df, 'BsmtUnfSF')
df = add_percentile_column(df, 'TotalBsmtSF')
df = add_percentile_column(df, '1stFlrSF')
df = add_percentile_column(df, '2ndFlrSF')
df = add_percentile_column(df, 'LowQualFinSF')
df = add_percentile_column(df, 'BsmtFullBath')
df = add_percentile_column(df, 'BsmtHalfBath')
df = add_percentile_column(df, 'FullBath')
df = add_percentile_column(df, 'HalfBath')
df = add_percentile_column(df, 'BedroomAbvGr')
df = add_percentile_column(df, 'KitchenAbvGr')
df = add_percentile_column(df, 'TotRmsAbvGrd')
df = add_percentile_column(df, 'Fireplaces')
df = add_percentile_column(df, 'GarageYrBlt')
df = add_percentile_column(df, 'GarageCars')
df = add_percentile_column(df, 'GarageArea')
df = add_percentile_column(df, 'WoodDeckSF')
df = add_percentile_column(df, 'OpenPorchSF')
df = add_percentile_column(df, 'EnclosedPorch')
df = add_percentile_column(df, '3SsnPorch')
df = add_percentile_column(df, 'ScreenPorch')
df = add_percentile_column(df, 'PoolArea')
df = add_percentile_column(df, 'MiscVal')
df = add_percentile_column(df, 'MoSold')
df = add_percentile_column(df, 'YrSold')

# null checking

null_val=df.isna().sum()
df['LotFrontage_percentile'] = df['LotFrontage_percentile'].fillna(0)
df['GarageYrBlt_percentile'] = df['GarageYrBlt_percentile'].fillna(0)
df['MasVnrArea_percentile'] = df['MasVnrArea_percentile'].fillna(0)

null_New_val=df.isna().sum()

df.to_csv("C:\\Users\\doron\\OneDrive\\שולחן העבודה\\פרויקט גמר\\train_ready_for_ANN.csv")


