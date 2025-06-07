#!/bin/bash
out="setup.ini"
> $out

for folder in ath bol aar tha
do
echo "genome=${folder}" >> $out
ls $folder > "${folder}_files.txt"
awk -v folder=$folder '{split($0, a, "."); print a[1] " data/" folder "/" $0}' "${folder}_files.txt" >> $out
echo -en "\n" >> $out
rm ${folder}_files.txt
done