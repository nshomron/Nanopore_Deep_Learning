#!/bin/tcsh
#$ -N poretools
#$ -S /bin/tcsh
#$ -j y
#$ -e $JOB_NAME.$JOB_ID.ERR
#$ -cwd
#$ -l nsh

set NCORES=`grep ^processor /proc/cpuinfo | wc -l`
@ NCORES= ( $NCORES - 1 )

## change to the correct flowcell name
set FLOWCELLNAME="20180812_1318_12Aug_hek3_noSelection_2Hours"


mkdir -p fastq

## change  to the correct location of poretools script
 ~/.local/bin/poretools fastq --min-length 200 basecalls/$FLOWCELLNAME/workspace/pass/ > fastq/$FLOWCELLNAME.fastq
 