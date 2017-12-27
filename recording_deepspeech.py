import os
import datetime

recorded = []
for dn, _, fs in os.walk("data/self/"):
    name = dn.split("/")[1]
    if name:
        recorded.append(name)

with open("data/self/recording_scripts.txt", "r") as f:
    scripts = f.readlines()

# record one command by 20 times
scripts_20 = []
for idx, src in enumerate(scripts):
    src_20 = [(idx, src.replace("\n", ""))] * 20
    scripts_20.extend(src_20) 

person = input("Enter your initials: (e.g. CL, WH, BZ...) ")

while True:
    if person in recorded:
        print("{} already exists. Please enter another name.".format(person))
        person = input("Enter your initials: (e.g. CL, WH, BZ...) ")
    else:
        break

os.mkdir("data/self/" + person)
person_folder = "data/self/" + person + "/"

for count, value in enumerate(scripts_20):
    idx = value[0]
    script = value[1]
    filename = str(datetime.date.today()).replace("-", '') + "_" + person + "_" + str(count+1) + "_" + str(idx) + ".wav"
    print("-" * 60 + "\n{}: {}\n".format(count+1, script))
    input("Press ENTER and start recording >>> ")
    os.system("arecord -d 5 -r 16000 -f S16_LE " + person_folder + filename)

