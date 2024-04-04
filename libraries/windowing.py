import pandas as pd

def return_windows(df, window_duration=3, data_frequency=25, min_window_size=0.95, overlap=0, datetime_column_name='DateTime'):
    
    """
    Extracts windows of data from a time-series DataFrame.

    Parameters:
    - df (pandas DataFrame): Input DataFrame containing time-series data.
    - window_duration (int or float): Duration of each window in seconds.
    - data_frequency (int or float): Frequency of the data within the main DataFrame(df).
    - min_window_size (int or float) [0-1]: Minimum amount of data a window must have (% of a full window size(1)).
    - overlap (float) [0-1]: Overlap between consecutive windows as a percentage.
    - datetime_column_name (str): Name of the column with datetime information.

    Returns:
    - list: A list of DataFrame windows extracted based on the provided parameters.

    Raises:
    - ValueError: If the input DataFrame is empty.
    """
    
    if len(df) > 0:
        
        df = df.sort_values(by=datetime_column_name)
        df = df.reset_index(drop=True)
        
        start_time = df[datetime_column_name].iloc[0]
        end_time = start_time + pd.Timedelta(seconds=window_duration)
        
        window_size_limit = data_frequency*window_duration*min_window_size
        
        windows = []
        
        while(True):
            df_window = df[(df[datetime_column_name] >= start_time) & (df[datetime_column_name] < end_time)]
            
            if len(df_window) >= window_size_limit:
                df_window = df_window.reset_index(drop=True)
                windows.append(df_window)
                
            start_time = end_time - pd.Timedelta(seconds=window_duration*overlap)
            end_time = start_time + pd.Timedelta(seconds=window_duration)

            if(start_time >= df[datetime_column_name].iloc[-1]):
                break

        return windows
    
    else:
        raise ValueError('Empty dataframe!')
        return []