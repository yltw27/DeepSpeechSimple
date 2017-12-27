"""run .py in data/librivox/"""

import os
from datetime import datetime


def flac2wav(folder):
    start_point = datetime.now()
    count_files = 0
    for dirpath, _, files in os.walk(folder):
#        print(dirpath)
        for a_file in files:
            if ".flac" in a_file:
                count_files += 1
#                print(dirpath, a_file, a_file.replace(".flac", ".wav"))
                os.system("sox " + dirpath + "/" + a_file + " " + dirpath + "/" + a_file.replace(".flac", ".wav"))
    print("folder: {}, data: {}\nprocessing time: {}".format(folder, count_files, datetime.now()-start_point))


#if __name__ == "__main__":
##    flac2wav("dev-clean/")
##    flac2wav("train-clean-100/")
##    flac2wav("train-clean-360/")
##    flac2wav("train-other-500/")
#    flac2wav("test-clean")

