import sys
import os
from glob import glob
import re

def processAll(path):
    txt_files = glob(os.path.join(path, "*.trans.txt"))
    if(len(txt_files)>0):
        with open(txt_files[0]) as f:
            line = f.readline()
            while line:
                name=re.findall(r'\d+-\d+-\d+',line)
                filename=path+'/'+name[0]+'.txt'
                with open(filename,"w") as ff:
                    ff.write(line[len(name[0])+1:].lower())
                line = f.readline()
        os.remove(txt_files[0])
    parents = os.listdir(path)
    for parent in parents:
        child = os.path.join(path,parent)
        if os.path.isdir(child):
            processAll(child)

processAll(sys.argv[1])
