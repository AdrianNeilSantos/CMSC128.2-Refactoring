from sklearn.preprocessing import LabelEncoder, StandardScaler
from pandas import DataFrame

def remove_duplicates():
    pass

def recode_categorical(data_frame, categorical_cols):
    number = LabelEncoder()
    for col in categorical_cols:
        data_frame[col] = number.fit_transform(data_frame[col].astype(str))
    return data_frame

def standard_scale(data_frame, df_nodup, target):
    orig_cols = list(data_frame.columns)

    # Let's apply StandardScaler() to the dataset with no outliers
    trans = StandardScaler()
    data_frame = trans.fit_transform(data_frame)

    # convert the array back to a dataframe
    data_frame = DataFrame(data_frame)
    # reassign the column names
    data_frame.columns = orig_cols

    data_frame[target] = df_nodup[target]

    return data_frame