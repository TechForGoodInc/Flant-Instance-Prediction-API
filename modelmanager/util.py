import pickle
import logging
import os
from datetime import datetime
import pandas as pd
from modelmanager.constants import Constants
import csv


def get_data(
        ec2, 
        start_time,
        end_time=datetime.now(),
        instance_types=[],
        product_descriptions=[],
        availability_zones='',
        max_results=0):

    if type(start_time) == str:
        start_time = datetime.strptime(start_time, "%m-%d-%y")
    if type(end_time) == str:
        end_time = datetime.strptime(end_time, "%m-%d-%y")
    
    # startTime = datetime.datetime.now() - datetime.timedelta(seconds=Seconds, hours=Hours, days=Days, weeks=Weeks)
    spot_price_history = []
    next_token = '0'

    while len(next_token) > 0 and (max_results == 0 or len(spot_price_history) < max_results):
        res = ec2.describe_spot_price_history(
            StartTime=start_time,
            EndTime=end_time,
            InstanceTypes=instance_types,
            ProductDescriptions=product_descriptions,
            AvailabilityZone=availability_zones,
            MaxResults=max_results if max_results > 0 else 1000,
            NextToken=next_token if next_token != '0' else '')

        spot_price_history += res['SpotPriceHistory']
        next_token = res['NextToken']

    return pd.DataFrame(spot_price_history)


# Pickle Object
def save_obj(obj, file_name, path=Constants.OBJ_PATH):
    if not os.path.exists(path):
        os.makedirs(path)
    with open(path + file_name + '.pkl', 'wb') as f:
        pickle.dump(obj, f)


# Load pickled object
def load_obj(file_name, type=list, path=Constants.OBJ_PATH):
    try:
        with open(path + file_name + '.pkl', 'rb') as f:
            return pickle.load(f)
    except (IOError, EOFError) as e:
        return type()


# Save data frame to csv
def save_df(file_name, path, df, index_label=None):
    if not os.path.exists(path):
        os.makedirs(path)
    with open(path + file_name + '.csv', 'w') as f:
        df.to_csv(f, index_label=index_label, line_terminator='\r')


product_desc_table = {
    'Red Hat Enterprise Linux': 'RL',
    'SUSE Linux': 'SL',
    'Linux/UNIX': 'LU',
    'Windows': 'WI'
}

for key, val in dict(product_desc_table).items():
    product_desc_table[val] = key


def encode_name(av_zone, ins_type, pro_desc):
    return str(av_zone + '_' + ins_type + '_' + product_desc_table[pro_desc])


def decode_name(encoded_name):
    split = str(encoded_name).split('_')

    if len(split) != 3:
        raise ValueError('Improper string passed to decodeName')

    return split[0], split[1], product_desc_table[split[2]]
