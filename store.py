import os
import json, csv, pickle, xlsxwriter, xlrd
import pandas as pd

f_jn = lambda folder, file: os.path.join(folder, file)

def saveFile(data, folder, file):
    filepath = f_jn(folder, file)

    if (file[-4:].lower() == 'json'):
        saveJSON(data, filepath)
    elif (file[-4:].lower() == 'xlsx'):
        saveXLSX(data, filepath)
    elif (file[-3:].lower() == 'txt'):
        saveTXT(data, filepath)
    elif (file[-3:].lower() == 'csv'):
        saveCSV(data, filepath)
    elif (file[-3:].lower() == 'pkl'):
        savePKL(data, filepath)
    elif (file[-4:].lower() == 'xlsx'):
        saveXLSX(data, filepath)
    elif (file[-3:].lower() == 'png'):
        data.savefig(filepath)

    return None


def loadFile(folder, file):
    filepath = f_jn(folder, file)

    if (file[-4:].lower() == 'json'):
        return loadJSON(filepath)
    elif (file[-4:].lower() == 'xlsx'):
        return loadXLSX(filepath)
    elif (file[-3:].lower() == 'txt'):
        return loadTXT(filepath)
    elif (file[-3:].lower() == 'csv'):
        return loadCSV(filepath)
    elif (file[-3:].lower() == 'pkl'):
        return loadPKL(filepath)

    return None


def saveCSV(data, filepath):
    with open(filepath, 'w', newline='', encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerows(data)

def loadCSV(filepath):
    data = []
    with open(filepath, newline='', encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            data.append(row)
    return data

def saveJSON(data, filepath):
    with open(filepath, 'w', encoding='utf8') as outfile:
        json.dump(data, outfile, sort_keys=True, indent=4, separators=(',', ': '), ensure_ascii=False)

def loadJSON(filepath):
    try:
        return json.loads(''.join(loadTXT(filepath)))
    except:
        return {}

def saveTXT(data, filepath):
    with open(filepath, 'r') as f:
        for line in data:
            f.write(line)

def loadTXT(filepath):
    strList = []
    infile = open(filepath, encoding="utf-8")
    for line in infile:
        strList.append(line)
    return strList

def savePKL(data, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)

def loadPKL(filepath):
    with open(filepath, 'rb') as f:
        try:
            model = pickle.load(f, encoding='latin1')
        except:
            model = pickle.load(f)
    return model

def saveXLSX(data, filepath):

    def writeSheet(sheet, data):
        for i in range(len(data)):
            for j in range(len(data[i])):
                sheet.write(i, j, data[i][j])

    workbook = xlsxwriter.Workbook(filepath)
    for shName, sh in data.items():
        writeSheet(workbook.add_worksheet(shName), sh)
    workbook.close()


def loadXLSX(filepath, sheet_Ids=[0]):
    data = []

    for sid in sheet_Ids:
        tmpsheet = xlrd.open_workbook(filepath).sheet_by_index(0)
        for i in range(tmpsheet.nrows):
            data.append(tmpsheet.row_values(i))

    return data

def getFileList(folder):
    return [f for f in os.listdir(folder) if os.isfile(f_jn(folder, f))]


def saveDataFrame(data, folder, file):
    data.to_csv(f_jn(folder, file), header=None, index=None)

def loadDataFrame(folder, file):
    return pd.read_csv(f_jn(folder, file), header=None)