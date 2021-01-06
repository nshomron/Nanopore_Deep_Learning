#!/bin/tcsh


set IFS='\n'
foreach f(`ls ./separating_files/*.readIDList.txt \
`)
    set FILENAME=`echo "$f" | sed "s/\.\/separating_files\///g"`
    qsub   ./runGetLinksAndDownloadBasedOnReadIDListFiles.sh $f $FILENAME
end
