def perform_eda(data_frame, head=True, info=True, describe=True):
    if head:
       data_frame.head()
    if info:
       data_frame.info()
    if describe:
       data_frame.describe()
