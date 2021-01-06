#!/bin/tcsh
#$ -N runDownload
#$ -S /bin/tcsh
#$ -j y
#$ -o $JOB_NAME.$JOB_ID.OUT
#$ -cwd
#$ -l nsh

#var $1 - filepath of readIDlist file
#var $2 - filename of readIDlist file$

set DIR=`pwd`
set FILENAME=`echo "$2" | sed "s/.readIDList.txt//g"`
setenv LC_ALL C

echo "Working on file: " $FILENAME
echo "Location of file is: " $DIR/$1

mkdir -p $DIR/separating_files/$FILENAME
mkdir -p $DIR/separating_files/$FILENAME/fast5Links

./selectLinesFromF2BasedOnF1.sh $DIR/$1 $DIR/separating_files/concat_seq_summary.txt |\
 sed '/^V1/ d' > $DIR/separating_files/$FILENAME/selectedLinesFromBam.txt

cat $DIR/separating_files/$FILENAME/selectedLinesFromBam.txt |\
 awk '{print $1}' > $DIR/separating_files/$FILENAME/selectedFast5Filenames.txt



foreach f(`cat $DIR/separating_files/$FILENAME/selectedFast5Filenames.txt `)
    #wget -P $DIR/fast5Download_FinalOut_fast5/$FILENAME/ "$f"
    ## change location to where the links to the fast5 files, in this example its "linksToAllFast5"
    ln -s $DIR/linksToAllFast5/$f $DIR/separating_files/$FILENAME/fast5Links/
end

echo "Getting CSV files ready"
python ./get_signal.py --pathOfFast5Files separating_files/$FILENAME/fast5Links/ --prefixName $FILENAME
echo "Finished getting CSV files ready"


echo "Finished script"
