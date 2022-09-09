import sys
import os

def convert(pdf):
    print "converting"
    filename = os.path.basename(pdf)[0:-4]
    os.system("convert -density 150 %s -quality 90 %s.png"%(pdf, filename))

def main():

    pdf = sys.argv[1]

    if pdf[-3:] != "pdf":
        print "input file is not a pdf"
    else:
        convert(pdf)

if __name__ == '__main__':
    main()
