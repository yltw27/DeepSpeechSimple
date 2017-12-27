"""
run .py from DeepSpeech/
python3 data/LibriVox/catalog.py
"""

import os
import csv
from datetime import datetime


def get_script(txt_file):
    """return a list of tuples (index of wav, script in lowercase)"""
    with open(txt_file, 'r') as f:
        scripts = f.readlines()
    
    result = []
    for script in scripts:
        first_blank = script.find(' ')    
        idx = script[:first_blank]
        sentence = script[first_blank+1:].lower().replace('\n', '')
        result.append((idx, sentence))
    return result


def create_catalog(folder_list, csv_filename):
    """
    1. read each txt in each folder (path from DeepSpeech)
    2. find the information of wav with the certain index
    3. create a csv catalog with ['wav_filename', 'wav_filesize', 'transcript']
    """
    start_time = datetime.now()
    data = []
    
    for folder in folder_list:
        for dirpath, _, filenames in os.walk(folder):
            for filename in filenames:
                if '.trans.txt' in filename:
                    script_list = get_script(dirpath + '/' + filename)
                    for idx, script in script_list:
                        wav_filename = dirpath + '/' + idx + '.wav'
                        wav_filesize = os.path.getsize(wav_filename)
                        data.append((wav_filename, wav_filesize, script))
    print('data in {}: {}'.format(folder_list, len(data)))
    
    with open(csv_filename, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['wav_filename', 'wav_filesize', 'transcript'])
        writer.writerows(data)
    print('csv file created: {}\ntime: {}'.format(csv_filename, datetime.now()-start_time))


create_catalog(folder_list=["data/LibriVox/dev-clean/"], csv_filename="data/LibriVox/dev-clean.csv")
create_catalog(folder_list=["data/LibriVox/test-clean/"], csv_filename="data/LibriVox/test-clean.csv")
create_catalog(folder_list=["data/LibriVox/train-clean-100/",
                            "data/LibriVox/train-clean-360/",
                            "data/LibriVox/train-other-500/"], csv_filename="data/LibriVox/train-all.csv")


