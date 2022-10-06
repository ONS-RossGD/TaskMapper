import csv
import json
from typing import List, Tuple
import pandas as pd
from glob import glob

CONFIG_FILE_PATH = r'Inventories Local_Config_Report_INT23_22_Sep_2022'


def group_paths() -> List:
    """Returns list of filepaths to group config csvs"""
    groups = []
    for path in glob(f'{CONFIG_FILE_PATH}/*.csv'):
        file = path.split('\\')[-1]
        if file[:3] == "Grp" and file[:4] != "Grps":
            groups.append(f'{CONFIG_FILE_PATH}/{file}')
    return groups


def convert_group(path: str) -> Tuple[str, str, List[str]]:
    """Converts a CSV group config file to provide it's classification
    as a string, the group name as a string, and the items as a list of strings"""
    with open(path) as file:
        reader = csv.reader(file)
        next(reader)
        info = next(reader)[0]
    split_info = info.split('Classification: ')[-1].split(" Group: ")
    classification = split_info[0].strip()
    group = split_info[1].strip()
    df = pd.read_csv(path, skiprows=3, encoding='unicode_escape')
    items = df['Code'].tolist()
    print(classification, group, items)
    return classification, group, items


if __name__ == "__main__":
    groups = group_paths()
    master_dict = dict()
    for group in groups:
        classification, group_name, items = convert_group(group)
        try:
            master_dict[classification].update({group_name: items})
        except KeyError:
            master_dict[classification] = {}
            master_dict[classification].update({group_name: items})
    with open('groups.json', 'w') as file:
        json.dump(master_dict, file)
