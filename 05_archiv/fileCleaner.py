# read all files and check line by line if data is ok, if not print filetitle

# run from develop directory
import re
import os
#------------------------------FUNCTIONS--------------------------------------------------
def commaFixer(filePath):
    print('Fixing file')
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
    print('Rplace "," with ";"')
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
    print('Fixing Date')
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

for root, dirs, files in os.walk(f"./data_temp/collectorDuka/"):
    for file in files:
        filePath = os.path.join(root, file)
        with open(filePath) as fp:
            lines = fp.readlines()
            lineCount = 0
            for line in lines:
                lineCount+=1
                if re.search(completlyBrokenPattern, line):
                    print('ERROR - FILE BROKEN AF:')
                    print(filePath)
                    writeToBrokenFiles(file)
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