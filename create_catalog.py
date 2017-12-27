import os
import csv

filenames = []
for d, ds, fs in os.walk('data/train/'):
    for f in fs:
        if '.wav' in f:
            filenames.append(d+'/'+f)

with open('data/train/train.txt', 'r') as f:
    scripts = f.readlines()

scripts = scripts * 100

training_data = []
for i in range(len(filenames)):
    if 'A01' in filenames[i]:
        script = scripts[int(filenames[i].split('_A01_')[-1][:-4])-1]
    else:
        script = scripts[int(filenames[i].split('voice')[-1][:-4])-1]
    if script[-1] == '\n':
        script = script[:-1]
    size = os.path.getsize(filenames[i])
    training_data.append((filenames[i], size, script))

with open('data/train/train.csv', 'w') as f:
    writer = csv.writer(f)
    header = ['wav_filename', 'wav_filesize', 'transcript']
    writer.writerow(header)
    writer.writerows(training_data)

