import csv
import sys


def process_header_row(header_row):
    header = []
    process_columns = []
    for index, col in enumerate(header_row):
        if index < 4 or "_labels" in col:
            col_name = col.replace("_labels", "").strip().replace(" ", "_")
            # print(f"{index}:'{col_name}'")
            process_columns.append(index)
            header.append(col_name)
    return header, process_columns


def load_row(process_columns, row):
    values = []
    for index in process_columns:
        values.append(row[index])
    return values


def load_survey(survey_file, max_rows=sys.maxsize):
    with open(survey_file) as f:
        csv_reader = csv.reader(f)

        header_row = next(csv_reader)
        header, proccess_columns = process_header_row(header_row)
        header_extra_row = next(csv_reader)
        row_number = 0
        value_rows = []
        for row in csv_reader:
            year = float(row[0])
            if year < 2000 or year > 2012:
                continue
            values = load_row(proccess_columns, row)
            value_rows.append(values)
            row_number += 1
            print(row_number)
            if row_number > max_rows:
                break
    return header, value_rows


def write_updated_survey(csv_file, header, value_rows):
    with open(csv_file, "w") as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)

        # writing the fields
        csvwriter.writerow(header)

        # writing the data rows
        csvwriter.writerows(value_rows)


survey_csv = "data/survey/gss.csv"

header, value_rows = load_survey(survey_csv)

new_csv = "data/survey/gss_modified.csv"

write_updated_survey(new_csv, header, value_rows)
