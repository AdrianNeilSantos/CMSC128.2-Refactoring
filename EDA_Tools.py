def describe_df(data_frame, head=False, info=False, describe=False):
    if head:
       data_frame.head()
    if info:
       data_frame.info()
    if describe:
       data_frame.describe()

def count_col_value(data_frame, column_name, withPercentage=False):
    data_frame[column_name].value_counts()
    if withPercentage:
        data_frame[column_name].value_counts(normalize=True)*100

