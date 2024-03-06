"""
my_data_tools: A Python library for data manipulation and analysis.

This library provides various functions for data analysis, manipulation, and normalization.

Functions:
- `add_one(number)`: Adds one to the given number.
- `check_ip()`: Prints the user's IP address.
- `data_index(extension, csv_files)`: Generates an index for CSV files in a specified folder.
- `dogcat(human_years)`: Converts human years to cat and dog years.
- `fill_nulls_with_mean(dataframe, column)`: Fills null values in the specified column with the mean.
- `fill_nulls_with_median(dataframe, column)`: Fills null values in the specified column with the median.
- `help()`: Prints the available functions in the library.
- `info(key)`: Provides information about a specific function.
- `normalise_column(dataframe, column)`: Normalizes the specified column in the dataframe.
- `outliers_IQR(dataframe, column)`: Calculates and prints the lower and upper bounds using IQR method.
- `remove_outliers(dataframe, column, lower_threshold, upper_threshold)`: Removes outliers in the specified column based on the given thresholds.
- `remove_special_char(df, column_name)`: Removes special characters from the specified column in the dataframe.
- `split_datetime_column(df, column_name)`: Splits a datetime column into separate components such as day, month, year, and time.
- `split_string(x)`: Splits a string into numerical and alphabetical characters.
- `split_string_column(df, column)`: Splits a string column into two new columns containing alphabetical and numerical parts.
- `split_time(df, column)`: Splits a time column into hours, minutes, and seconds.
- `SQL_table_format_from_data_index(Datadict, specified_table)`: Generates SQL table format from the data index for the specified table.
- `standardise_column(dataframe, column)`: Standardizes the specified column in the dataframe.
- `standardise_date_format(dataframe, date_column)`: Standardizes the date format in the specified date column.

For more information on each function, use `info(function_name)`.
"""

import pandas as pd
import numpy as np
import re
from datetime import datetime
from sklearn import preprocessing  
import requests

def add_one(number):
    """
    Adds one to the given number.
    """
    try:
        return number + 1
    except TypeError:
        print("Error: Input must be a numeric value.")

def check_ip():
    """
    Prints the user's IP address.
    """
    try:
        response = requests.get('https://httpbin.org/ip')
        print('Your IP is {0}'.format(response.json()['origin']))
    except requests.RequestException as e:
        print(f"Error fetching IP: {e}")

def data_index(extension, csv_files):
    """
    Generates an index for CSV files in a specified folder.
    """
    newlist = []
    try:
        for csv in csv_files:
            temp_list = []
            try:
                df_tmp = pd.read_csv(extension + csv, encoding='latin1')
                
                # Extract column information
                column_info = df_tmp.dtypes.reset_index()
                column_info.columns = ['Column', 'DataType']

                # Add metadata to the column information
                column_info['Table name'] = csv.split('.')[0]
                column_lengths = df_tmp.count().values
                column_info['Column Length'] = column_lengths
                column_info['Table Length'] = len(df_tmp)

                temp_list.append(column_info)
                newlist.append(temp_list)
            except pd.errors.EmptyDataError:
                print(f"Warning: Empty file - {csv}")

        # Concatenate the dataframes into a single data dictionary
        Datadict = pd.concat([df for sublist in newlist for df in sublist])
        cols = ['Table name', 'Column', 'DataType', 'Column Length', 'Table Length']
        Datadict = Datadict[cols]
        return Datadict
    except pd.errors.ConcatenateError:
        print("Error: Unable to concatenate dataframes.")

def dogcat(human_years):
    """
    Converts human years to cat and dog years.
    """
    cat_years = 0
    dog_years = 0
    x = 1
    try:
        for i in range(human_years):
            if x == 1:
                cat_years += 15
                dog_years += 15
                x += 1
                continue
            if x == 2:
                cat_years += 9
                dog_years += 9
                x += 1
                continue
            cat_years += 4
            dog_years += 5

        # Print the results
        print(['Number of human years:', human_years, 'Number of cat years:', cat_years, 'Number of dog years:', dog_years])
    except TypeError:
        print("Error: Input must be a numeric value.")

def fill_nulls_with_mean(dataframe, column):
    """
    Fills null values in the specified column with the mean.
    """
    try:
        mean = dataframe[column].mean()
        dataframe[column].fillna(mean, inplace=True)
    except KeyError:
        print(f"Error: Column '{column}' not found in the dataframe.")

def fill_nulls_with_median(dataframe, column):
    """
    Fills null values in the specified column with the median.
    """
    try:
        median = dataframe[column].median()
        dataframe[column].fillna(median, inplace=True)
    except KeyError:
        print(f"Error: Column '{column}' not found in the dataframe.")

def help():
    """
    Prints the available functions in the library.
    """
    try:
        print("def check_ip()")
        print("data_index(extension,csv_files)")
        print("dogcat(human_years)")
        print("fill_nulls_with_mean(dataframe, column)")
        print("fill_nulls_with_median(dataframe, column)")
        print("normalise_column(dataframe, column)")
        print("outliers_IQR(dataframe,column)")
        print("remove_outliers(dataframe, column, lower_threshold, upper_threshold)")
        print("remove_special_char(df, column_name)")
        print("split_string(string)")
        print("split_string_column(df, column)")
        print("split_datetime_column(df, column_name)")
        print("split_time(df, column)")
        print("SQL_table_format_from_data_index(data_index,specified_table)")
        print("standardise_column(dataframe, column)")
        print("standardise_date_format(dataframe, date_column)")
        print("For more help place the function name into the function info, example = cd.info(dogcat)")
    except Exception as e:
        print(f"Error: {e}")

def info(key):
    """
    Provides information about a specific function.
    """
    try:
        info_dict = {'split_time':'split_time(df, column), takes a column with time data and splits it into hour, minute, and second',
                     'check_ip':'def check_ip(), this function will return your ip',
                     'split_string_column':'split_string_column(df, column), splits the string of a column into the alphabetical and numerical parts, works best for ID format of MD1996',
                     'split_datetime_column':'split_datetime_column(df, column_name), this takes a datetime column from a df and spits it into day, month, year, and time if its present', 
                     'standardise_date_format':'standardise_date_format(dataframe, date_column), takes a dataframe and the column with the dates, takes the first format it finds and formats the whole column to the same date format',
                     'data_index':'data_index(extension,csv_files) the extension is the pathway to the folder with the files, csv_files is a list of the csv names to be part of the index',
                     'dogcat':' dogcat(human_years): tells you the cat and dog ages compared to human years',
                     'split_string': 'split_string(string): spilts a string into numerical and alphabetical characters',
                     'outliers_IQR': "outliers_IQR(dataframe,column) takes the dataframe's column and returns the upper and lower bounds",
                     'normalise_column':'normalise_column(dataframe, column) normalises the selected column',
                     'fill_nulls_with_mean':'fill_nulls_with_mean(dataframe, column) fills the selected columns nulls with the columns mean',
                     'fill_nulls_with_median':'fill_nulls_with_median(dataframe, column fills the selected columns nulls with the columns median',
                     'remove_outliers':'remove_outliers(dataframe, column, lower_threshold, upper_threshold) having the upper and lower threshold found by using outliers_IQR removes outliers_IQR',
                     'remove_special_char':'remove_special_char(df, column_name), removes characters from the column of the dataframe used for £, commas,ect',
                     'standardise_column':'standardise_column(dataframe, column) takes a column from the dataframe and standardises',
                     'SQL_table_format_from_data_index':'SQL_table_format_from_data_index(data_index,specified_table), the data index is created with data_index,specified_table is the table name from the Table name column'}
        if key in info_dict:
            value = info_dict.get(key)
        return value
    except KeyError:
        print(f"Error: Function '{key}' not found in the info dictionary.")

def normalise_column(dataframe, column):
    """
    Normalizes the specified column in the dataframe.
    """
    try:
        x = dataframe[[column]].values
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        normalized_column = pd.DataFrame(x_scaled, columns=[column])
        dataframe[column] = normalized_column[column]
        return dataframe
    except KeyError:
        print(f"Error: Column '{column}' not found in the dataframe.")

def outliers_IQR(dataframe, column):
    """
    Calculates and prints the lower and upper bounds using IQR method.
    """
    try:
        lower = np.percentile(dataframe[column], 25)
        upper = np.percentile(dataframe[column], 75)
        IQR = (upper - lower)
        lower_b = (lower) - (1.5 * IQR)
        upper_b = (upper) + (1.5 * IQR)
        return print('the lower bound is:', lower_b, 'the upper bound is:', upper_b)
    except KeyError:
        print(f"Error: Column '{column}' not found in the dataframe.")

def remove_outliers(dataframe, column, lower_threshold, upper_threshold):
    """
    Removes outliers in the specified column based on the given thresholds.
    """
    try:
        dataframe.drop(dataframe[(dataframe[column] < lower_threshold) | 
                                (dataframe[column] > upper_threshold)].index, inplace=True)
    except KeyError:
        print(f"Error: Column '{column}' not found in the dataframe.")

def remove_special_char(df, column_name):
    """
    Removes special characters from the specified column in the dataframe.
    """
    try:
        df[column_name] = df[column_name].apply(lambda string: ''.join(e for e in string if e.isalnum()))
        return df
    except KeyError:
        print(f"Error: Column '{column_name}' not found in the dataframe.")

def split_datetime_column(df, column_name):
    """
    Splits a datetime column into separate components such as day, month, year, and time.
    """
    try:
        if column_name not in df.columns:
            print(f"Column '{column_name}' does not exist in the DataFrame.")
            return df

        datetime_format = pd.to_datetime(df[column_name], errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')

        has_time = datetime_format.str.contains('\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}').any()

        if has_time:
            df['Date'] = pd.to_datetime(df[column_name]).dt.date
            df['Day'] = pd.to_datetime(df[column_name]).dt.day
            df['Month'] = pd.to_datetime(df[column_name]).dt.month
            df['Year'] = pd.to_datetime(df[column_name]).dt.year
            df['Time'] = pd.to_datetime(df[column_name]).dt.time
        else:
            df['Day'] = pd.to_datetime(df[column_name]).dt.day
            df['Month'] = pd.to_datetime(df[column_name]).dt.month
            df['Year'] = pd.to_datetime(df[column_name]).dt.year

        return df
    except pd.errors.OutOfBoundsDatetime:
        print(f"Error: Out of bounds datetime in column '{column_name}'.")

def split_string(x):
    """
    Splits a string into numerical and alphabetical characters.
    """
    try:
        if x[0].isdigit():
            match = re.search(r"([0-9]+)([a-zA-Z]+)", x)
        else:
            match = re.search(r"([a-zA-Z]+)([0-9]+)", x)
        if match:
            return match.groups()
        else:
            return None
    except (IndexError, TypeError):
        print("Error: Invalid input for string splitting.")

def split_string_column(df, column):
    """
    Splits a string column into two new columns containing alphabetical and numerical parts.
    """
    try:
        def split_string(x):
            if x[0].isdigit():
                match = re.search(r"([0-9]+)([a-zA-Z]+)", x)
            else:
                match = re.search(r"([a-zA-Z]+)([0-9]+)", x)
            if match:
                df.loc[index, 'string1'] = match.group(1)
                df.loc[index, 'string2'] = match.group(2)
            else:
                df.loc[index, 'string1'] = None
                df.loc[index, 'string2'] = None

        df['string1'] = None
        df['string2'] = None
        for index, row in df.iterrows():
            split_string(row[column])
        return df
    except (KeyError, IndexError, TypeError):
        print(f"Error: Column '{column}' not found in the dataframe or invalid input for string splitting.")

def split_time(df, column):
    """
    Splits a time column into hours, minutes, and seconds.
    """
    try:
        def split_time(time_str):
            time_parts = time_str.split(':')
            if len(time_parts) == 3:
                return int(time_parts[0]), int(time_parts[1]), int(time_parts[2])
            elif len(time_parts) == 2:
                return int(time_parts[0]), int(time_parts[1]), 0
            elif len(time_parts) == 1:
                return int(time_parts[0]), 0, 0
            else:
                return None

        df[['Hour', 'Minute', 'Second']] = df[column].apply(split_time).apply(pd.Series)
    except (KeyError, ValueError):
        print(f"Error: Column '{column}' not found in the dataframe or invalid input for time splitting.")

def SQL_table_format_from_data_index(Datadict, specified_table):
    """
    Generates SQL table format from the data index for the specified table.
    """
    try:
        filtered_df = Datadict[Datadict['Table name'].str.contains(specified_table, case=False)]
        column_info_list = (filtered_df['Column'] + ' ' + filtered_df['DataType']).tolist()
        column_info_list = [item.strip("'") for item in column_info_list]
        column_info_list = [item.replace('int64', 'INT').replace('float64', 'FLOAT').replace('object', 'VARCHAR(255)').replace('bool', 'BOOLEAN') for item in column_info_list]
        for i in column_info_list:
            print(i)
    except KeyError:
        print(f"Error: Table '{specified_table}' not found in the data index.")

def standardise_column(dataframe, column):
    """
    Standardizes the specified column in the dataframe.
    """
    try:
        x = dataframe[[column]].values
        scaler = preprocessing.StandardScaler()
        x_scaled = scaler.fit_transform(x)
        standardized_column = pd.DataFrame(x_scaled, columns=[column])
        dataframe[column] = standardized_column[column]
        return dataframe
    except KeyError:
        print(f"Error: Column '{column}' not found in the dataframe.")

def standardise_date_format(dataframe, date_column):
    """
    Standardizes the date format in the specified date column.
    """
    try:
        last_record = dataframe[date_column].iloc[-1]  
        df = dataframe.append({date_column: last_record}, ignore_index=True) 
        first_date = dataframe[date_column].iloc[0]
        formats = ['%Y-%m-%d', '%m-%d-%Y', '%d-%m-%Y', '%Y/%m/%d', '%m/%d/%Y', '%d/%m/%Y']
        first_format = None

        for date_format in formats:
            try:
                datetime.strptime(first_date, date_format)
                first_format = date_format
                break
            except ValueError:
                continue

        standardised_dates = []
        for date in dataframe[date_column]:
            for date_format in formats:
                try:
                    standardised_dates.append(datetime.strptime(date, date_format).strftime('%Y-%m-%d'))
                    break
                except ValueError:
                    continue
        dataframe = dataframe[:-1]
        return pd.Series(standardised_dates)
    except ValueError:
        print(f"Error: Invalid date format in column '{date_column}'.")
