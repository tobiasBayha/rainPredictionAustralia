from pathlib import Path


def csv_data_path() -> Path:
    """
    Returns the location of the rain in Australia Dataset
    Relative path location to the file

    :return: the path to the CSV file
    """

    cwd = Path('')
    for folder in (cwd, cwd / '..'):
        data_folder = folder / 'data'
        if data_folder.exists() and data_folder.is_dir():
            #print('true')
            return data_folder / 'weatherAUS.csv'