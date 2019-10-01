#!/bin/bash
for FILE in $(ls test/)
do
	echo "Downsampling:" $FILE
	cat test/$FILE | python downsample.py 100 > downsampled/ds_$FILE
done
echo "Done."