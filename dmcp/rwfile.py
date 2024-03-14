import os
import sys
BITS=4
outfile="out_pytorch_int.txt"
fp=None

import os
def setbits(bits):
    global outfile
    global BITS
    BITS=bits
    outfile="out_pytorch_int{}.txt".format(BITS)

def openoutfile():
    global fp
    global outfile

    if os.path.exists(outfile):
        fp = open(outfile, 'a')
    else:
        fp = open(outfile, 'w')

def writeoutfile(x):
    fp.write(x)
def closeoutfile():
    fp.close()
def removeoutfile():
    if os.path.exists(outfile):
        os.remove(outfile)