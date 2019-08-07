# read all files and check line by line if data is ok, if not print filetitle
import sys
sys.path.append('01_preperation')

# run from develop directory
import re
import os
import time
import multiprocessing
from logger import *
folder = 'collectorDuka'

#------------------------------FUNCTIONS--------------------------------------------------
def commaFixer(filePath):
    sys.stdout.write(f"\r'Fixing file {os.path.basename(filePath)}\n")
    #read file and replce chars
    newlines = []
    with open(filePath) as fp:
        lines = fp.readlines()
        for line in lines:
            newLine = line.replace(',', '.')
            newlines.append(newLine)

    #delete old file
    os.remove(filePath)

    #write new file
    with open(filePath, "w") as f:
        for line in newlines:
            f.write(line)

def commaReplacer(filePath):
    sys.stdout.write(f"\r'Rplace ',' with ';' {os.path.basename(filePath)}\n")
    #read file and replce chars
    newlines = []
    with open(filePath) as fp:
        lines = fp.readlines()
        for line in lines:
            newLine = line.replace(',', ';')
            newlines.append(newLine)

    #delete old file
    os.remove(filePath)

    #write new file
    with open(filePath, "w") as f:
        for line in newlines:
            f.write(line)

def dateReplacer(filePath):
    sys.stdout.write(f"\r'Fixing Date {os.path.basename(filePath)}\n")
    #read file and replce chars
    newlines = []
    with open(filePath) as fp:
        lines = fp.readlines()
        for line in lines:
            newLine = line.replace('-', '.')
            newlines.append(newLine)

    #delete old file
    os.remove(filePath)

    #write new file
    with open(filePath, "w") as f:
        for line in newlines:
            f.write(line)           

def writeToBrokenFiles(file):
    with open("brokenFiles.txt", "a") as f:
        f.write(f'{file}\n')
#--------------------------------------------------------------------------------------

commaPattern = r'[0-9]{2}:[0-9]{2}:[0-9]{2};[0-9]+,'
completlyBrokenPattern = r"[0-9]{2}:[0-9]{2}:[0-9]{2},[0-9]+,"
commaReplacerMatch = r'[0-9]{2}:[0-9]{2}:[0-9]{2},'
dateMatch = r'[0-9]{4}-[0-9]{2}'

def checkfile(filePath):
    try:
        with open(filePath) as fp:
            lines = fp.readlines()
            lineCount = 0
            for line in lines:
                lineCount+=1
                if re.search(completlyBrokenPattern, line):
                    sys.stdout.write(f"\rERROR - FILE BROKEN AF: {os.path.basename(filePath)}\n")
                    writeToBrokenFiles(filePath)
                    lineCount=0
                    break
                if re.search(commaPattern, line):
                    fp.close()
                    commaFixer(filePath)
                    lineCount=0
                    break
                if re.search(commaReplacerMatch, line):
                    fp.close()
                    commaReplacer(filePath)
                    lineCount=0
                    break
                if re.search(dateMatch, line):
                    fp.close()
                    dateReplacer(filePath)
                    lineCount=0
                    break
                if lineCount > 4:
                    lineCount=0
                    break
    except:
        sys.stdout.write(f"\rCAN'T OPEN FILE: {os.path.basename(filePath)}\n")

def fileClearner():
    fileList = []
    for root, dirs, files in os.walk(os.path.join(os.getcwd(), "data_temp", "collectorDuka")):
        for file in files:
            fileList += [os.path.join(root, file)]

    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    result = pool.imap_unordered(checkfile, fileList)
    pool.close() # No more work
    while (True):
        completed = result._index
        sys.stdout.write(f'\r{Logger.bulk(completed, len(fileList))}    ')
        if (completed == len(fileList)): break
        time.sleep(0.1)
    sys.stdout.write('\n')

if __name__ == '__main__':
    fileClearner()