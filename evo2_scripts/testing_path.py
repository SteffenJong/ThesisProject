import sys

with open("paths.txt" , "w+") as file:
    file.write(str(sys.path))