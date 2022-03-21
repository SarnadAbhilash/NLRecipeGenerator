import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import platform
import time
import pathlib
import os
import json
import zipfile


# Load datasets from Json and preview them

def load_dataset(silent = False):
    # Download and get the path to downloaded dataset
    dataset_file_name = 'recipes_raw.zip'
    dataset_file_origin = 'https://storage.googleapis.com/recipe-box/recipes_raw.zip'

    dataset_file_path = tf.keras.utils.get_file(
        fname = dataset_file_name,
        origin = dataset_file_origin,
        cache_dir = cache_dir,
        extract = True,
        archive_format = 'zip'

    )

    print(dataset_file_path)

  dataset_file_names = [
        'recipes_raw_nosource_ar.json',
        'recipes_raw_nosource_epi.json',
        'recipes_raw_nosource_fn.json',                      
  ]
  dataset = []


  for dataset_file_name in dataset_file_names:
    dataset_file_path = f'{cache_dir}/datasets/{dataset_file_name}'

    
    with open(dataset_file_path) as dataset_file:
      json_data_dict = json.load(dataset_file)
      json_data_list = list(json_data_dict.values())
      dict_keys = [key for key in json_data_list[0]]
      dict_keys.sort()
      dataset += json_data_list


      if silent == False:
        print(dataset_file_path)
        print('=================================================')
        print('Number of examples: ', len(json_data_list), '\n')
        print('Example of object keys: ', dict_keys, '\n')
        print('Example of object: ', json_data_list[0], '\n')

        print('Required Keys: \n')
        print(' title: ', json_data_list[0]['title'], '\n')
        print(' ingredients: ', json_data_list[0]['ingredients'], '\n')
        print(' instructions: ', json_data_list[0]['instructions'], '\n')
        print('\n\n')

    

  return dataset





