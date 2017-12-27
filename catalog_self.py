import os
import csv

filenames = []
for d, ds, fs in os.walk('data/self/'):
    for f in fs:
        if '.wav' in f:
            filenames.append(d+'/'+f)

with open('data/self/recording_deepspeech.txt', 'r') as f:
    scripts = f.readlines()

training_data = []
for filename in range(filenames):
    
    idx = filename.split('_')[-1].replace('.wav', '')
    script = scripts[idx]
    
    if script[-1] == '\n':
        script = script.replace('\n', '')
    
    size = os.path.getsize(filenames)
    training_data.append((filenames, size, script))

with open('data/self/train_self.csv', 'w') as f:
    writer = csv.writer(f)
    header = ['wav_filename', 'wav_filesize', 'transcript']
    writer.writerow(header)
    writer.writerows(training_data)

