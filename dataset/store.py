import os
import csv
# import pandas as pd

def saveCSV(data, folder, file):
    with open(os.path.join(folder, file), 'w', newline='', encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerows(data)
    # df.to_csv(os.path.join(folder, file))

def loadCSV(folder, file):
    data = []
    with open(os.path.join(folder, file), newline='', encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            data.append(row)
    return data

