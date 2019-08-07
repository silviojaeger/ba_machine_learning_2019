from os import walk
import re

mypath = "U:\stocks_ch"

f = []
for (dirpath, dirnames, filenames) in walk(mypath):
    f.extend(filenames)
    break
first = True
for i in f:
    result = re.sub(r"_1 Min_Ask.*", "", i, 0)
    if first: 
        print(f"'{result}',", end =" ") 
        first = False
    else:
        first = True