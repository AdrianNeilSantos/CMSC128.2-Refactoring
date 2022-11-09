# Select features whose correlation with target is > 0.2
def get_features_by_corr(data_frame, target, corr_threshold):
    cor = data_frame.corr()
    cor_target = abs(cor[target])
    relevant_features = cor_target[cor_target >= corr_threshold]
    
    corr_columns_obj = relevant_features.index
    corr_columns_list = []
    
    for col in corr_columns_obj:
        if(col != target):
            corr_columns_list.append(col)

    return corr_columns_list
