import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import numpy as np


def get_column_names():
    column_names = []
    columns_to_encode = []
    with open("Assignment_1/data/census/column_names.txt") as f:
        line = f.readline()
        while line != "":
            start = line.find("(") + 1
            stop = line.find(")")
            column = line[start:stop]
            column_names.append(column)
            if "continuous" not in line:
                columns_to_encode.append(column)
            if column == "detailed household summary in household":
                column_names.append("instance weight")
                # columns_to_encode.append(False)
            line = f.readline()

    column_names.append("label")
    columns_to_encode.append("label")

    return column_names, columns_to_encode


def encode_dataframe(df, encode_columns, le):
    df_numeric = pd.DataFrame(columns=df.columns)

    for col in df.columns:
        if col in encode_columns:
            df_numeric[col] = le.transform(df[col])
        else:
            df_numeric[col] = df[col]

    return df_numeric


def create_data_and_label_dataframes(df, encode_columns, le):
    df_numeric = encode_dataframe(df, encode_columns, le)

    df_data = df.drop(columns=["instance weight", "label"])
    df_label = df["label"]

    df_data_numeric = df_numeric.drop(columns=["instance weight", "label"])
    df_label_numeric = df["label"].to_frame(name="label")
    df_label_numeric["label"] = np.where(df_label == " - 50000.", 0, 1)

    np_data_numeric = df_data_numeric.to_numpy()
    np_label_numeric = df_label_numeric.to_numpy()

    return df_data, df_label, np_data_numeric, np_label_numeric


def get_census_data_and_labels(scale_numeric):
    column_names, columns_to_encode = get_column_names()

    survey_train_csv = "Assignment_1/data/census/census-income.data"
    df_train = pd.read_csv(survey_train_csv, names=column_names, index_col=False)

    encode_columns = []
    for col in df_train.columns:
        if pd.api.types.is_string_dtype(df_train[col]):
            encode_columns.append(col)

    le = LabelEncoder()
    le = le.fit(df_train[encode_columns].to_numpy().flatten())

    df_train_data, df_train_label, np_train_data_numeric, np_train_label_numeric = create_data_and_label_dataframes(
        df_train, encode_columns, le
    )

    if scale_numeric:
        scaler = StandardScaler()
        scaler.fit(np_train_data_numeric)
        np_train_data_numeric = scaler.transform(np_train_data_numeric)

    survey_test_csv = "Assignment_1/data/census/census-income.test"
    df_test = pd.read_csv(survey_test_csv, names=column_names, index_col=False)
    df_test_data, df_test_label, np_test_data_numeric, np_test_label_numeric = create_data_and_label_dataframes(
        df_test, encode_columns, le
    )

    if scale_numeric:
        np_test_data_numeric = scaler.transform(np_test_data_numeric)

    return (
        df_train_data,
        df_train_label,
        np_train_data_numeric,
        np_train_label_numeric,
        df_test_data,
        df_test_label,
        np_test_data_numeric,
        np_test_label_numeric,
        le.classes_,
    )


def get_census_data_and_labels_one_hot():
    column_names, columns_to_encode = get_column_names()
    columns_to_encode.remove("label")

    survey_train_csv = "Assignment_1/data/census/census-income.data"
    df_train = pd.read_csv(survey_train_csv, names=column_names, index_col=False)
    df_train_one_hot = pd.get_dummies(df_train.drop(columns=["instance weight", "label"]), columns=columns_to_encode)
    np_train_one_hot = df_train_one_hot.to_numpy()
    scaler = MinMaxScaler()
    np_train_one_hot = scaler.fit_transform(np_train_one_hot)

    df_train_label = df_train["label"].to_frame(name="label")
    df_train_label["label"] = np.where(df_train_label == " - 50000.", 0, 1)
    np_train_label = df_train_label.to_numpy()

    survey_test_csv = "Assignment_1/data/census/census-income.test"
    df_test = pd.read_csv(survey_test_csv, names=column_names, index_col=False)
    df_test_one_hot = pd.get_dummies(df_test.drop(columns=["instance weight", "label"]), columns=columns_to_encode)
    df_test_one_hot = df_test_one_hot.reindex(columns=df_train_one_hot.columns, fill_value=0)
    np_test_one_hot = df_test_one_hot.to_numpy()
    np_test_one_hot = scaler.fit_transform(np_test_one_hot)

    df_test_label = df_test["label"].to_frame(name="label")
    df_test_label["label"] = np.where(df_test_label == " - 50000.", 0, 1)
    np_test_label = df_test_label.to_numpy()

    return (np_train_one_hot, np_train_label, np_test_one_hot, np_test_label)
