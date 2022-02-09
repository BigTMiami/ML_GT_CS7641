from Assignment_1.prep_census_data import get_column_names, get_census_data_and_labels

if __name__ == "main":
    column_names, columns_to_encode = get_column_names()
    CENSUS_DATA_LOCATION = "/Users/afm/Downloads/census/"
    (df_train_data,
        df_train_label,
        np_train_data_numeric,
        np_train_label_numeric,
        df_test_data,
        df_test_label,
        np_test_data_numeric,
        np_test_label_numeric,
        class_list) = get_census_data_and_labels(scale_numeric=True, CENSUS_DATA_LOCATION=CENSUS_DATA_LOCATION)
    np_train_data_numeric[0,:]
    np_train_label_numeric