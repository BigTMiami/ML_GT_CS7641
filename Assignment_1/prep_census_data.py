import pandas as pd
from sklearn.preprocessing import LabelEncoder


def get_column_names():
    column_names = []
    column_needs_encoding = []
    with open("Assignment_1/data/census/column_names.txt") as f:
        line = f.readline()
        while line != "":
            start = line.find("(") + 1
            stop = line.find(")")
            column = line[start:stop]
            column_names.append(column)
            if "continous" not in line:
                column_needs_encoding.append(column)
            if column == "detailed household summary in household":
                column_names.append("instance weight")
                # column_needs_encoding.append(False)
            line = f.readline()

    column_names.append("label")
    column_needs_encoding.append("label")

    return column_names, column_needs_encoding


def get_training_data_labels():
    column_names, column_needs_encoding = get_column_names()
    survey_csv = "data/census/census-income.data"

    df = pd.read_csv(survey_csv, names=column_names, index_col=False)

    encode_columns = []
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]):
            encode_columns.append(col)

    le = LabelEncoder()
    le = le.fit(df[encode_columns].to_numpy().flatten())

    df_numeric = pd.DataFrame(columns=df.columns)

    for col in df.columns:
        if col in encode_columns:
            df_numeric[col] = le.transform(df[col])
        else:
            df_numeric[col] = df[col]

    df_data = df.drop(columns=["instance weight", "label"])
    df_label = df["label"]

    df_data_numeric = df_numeric.drop(columns=["instance weight", "label"])
    df_label_numeric = df_numeric["label"]

    return df_data, df_label, df_data_numeric, df_label_numeric, le.classes_


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
    df_label_numeric = df_numeric["label"]

    return df_data, df_label, df_data_numeric, df_label_numeric


def get_census_data_and_labels():
    column_names, column_needs_encoding = get_column_names()

    survey_train_csv = "Assignment_1/data/census/census-income.data"
    df_train = pd.read_csv(survey_train_csv, names=column_names, index_col=False)

    encode_columns = []
    for col in df_train.columns:
        if pd.api.types.is_string_dtype(df_train[col]):
            encode_columns.append(col)

    le = LabelEncoder()
    le = le.fit(df_train[encode_columns].to_numpy().flatten())

    df_train_data, df_train_label, df_train_data_numeric, df_train_label_numeric = create_data_and_label_dataframes(
        df_train, encode_columns, le
    )

    survey_test_csv = "Assignment_1/data/census/census-income.test"
    df_test = pd.read_csv(survey_test_csv, names=column_names, index_col=False)
    df_test_data, df_test_label, df_test_data_numeric, df_test_label_numeric = create_data_and_label_dataframes(
        df_test, encode_columns, le
    )

    return (
        df_train_data,
        df_train_label,
        df_train_data_numeric,
        df_train_label_numeric,
        df_test_data,
        df_test_label,
        df_test_data_numeric,
        df_test_label_numeric,
        le.classes_,
    )
