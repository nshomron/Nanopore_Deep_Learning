#!/bin/tcsh
#$ -N albacore
#$ -S /bin/tcsh
#$ -j y
#$ -e $JOB_NAME.$JOB_ID.ERR
#$ -cwd
#$ -l nsh

set NCORES=`grep ^processor /proc/cpuinfo | wc -l`
@ NCORES= ( $NCORES - 1 )
## set the name of the flowcell as it was given during sequencing
set FLOWCELLNAME="20180809_1223_hek_run2_09Aug_moreThan0_99"


date
echo ""

## change to the correct location where ONT's albacore script is located
~/.local/bin/read_fast5_basecaller.py -r  --config r94_450bps_linear.cfg -k SQK-RAD004 -f FLO-MIN106  \
    -o fast5 -s ./ -t $NCORES -i ./fast5/$FLOWCELLNAME -s basecalls/$FLOWCELLNAME
set CURTIME=`date +"%m_%d_%T"`
cp basecalls/$FLOWCELLNAME/pipeline.log basecalls/$FLOWCELLNAME/$CURTIME.pipeline.log
cp basecalls/$FLOWCELLNAME/sequencing_summary.txt basecalls/$FLOWCELLNAME/$CURTIME.sequencing_summary.txt
echo ""
echo ">>> After basecalling"
date
echo ""

#    end
