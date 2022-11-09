import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


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


# create whitish correlation matrix
def correlation_matrix(df: pd.DataFrame):
    """
    A function to calculate and plot
    correlation matrix of a DataFrame.
    """
    # Create the matrix
    matrix = df.corr()

    # Create cmap
    cmap = sns.diverging_palette(250, 15, s=75, l=40, n=9, center="light", as_cmap=True)
    # Create a mask
    mask = np.triu(np.ones_like(matrix, dtype=bool))

    # Make figsize bigger
    fig, ax = plt.subplots(figsize=(25, 10))

    # Plot the matrix
    _ = sns.heatmap(
        matrix, mask=mask, center=0, annot=True, fmt=".2f", cmap=cmap, ax=ax
    )