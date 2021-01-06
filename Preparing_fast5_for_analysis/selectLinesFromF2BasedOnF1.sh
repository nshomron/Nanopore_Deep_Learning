## input should be:
## first file the paterns to search for
## second file is where to search the patterns


## with the clivome there is a problem where I have V1 and V2. So to solve this I filter out the V2 reads.
## What I do is I delete the lines beginning with "V1" from this oputput in the file runGetLinksAndDownloadBasedOnReadIDListFiles.sh

grep  -Fwf  $1 $2 
