#!/bin/tcsh
#$ -N minimap2
#$ -S /bin/tcsh
#$ -j y
#$ -o $JOB_NAME.$JOB_ID.OUT
#$ -cwd
#$ -l nsh


set DIR=`pwd`
set REFERENCEFASTA=$1
set PATHTOFASTQFILE=$2
set FASTQFILENAME=`echo "$2" | sed "s/.fastq//g"`
set NCORES=`grep ^processor /proc/cpuinfo | wc -l`
@ NCORES= ( $NCORES - 2 )


mkdir -p aln_statistics
mkdir -p separating_files
mkdir -p bam


    
## change to location of minimap2 instalation
echo "start of minimap2 mapping for $2"
~/tools/minimap2_23Jul/minimap2/minimap2 -ax map-ont $DIR/$REFERENCEFASTA  $DIR/$PATHTOFASTQFILE > $DIR/bam/$FASTQFILENAME.minimap2.sam
echo "end of minimap2 mapping for $2"

## cahnge to location of samtools if needed
echo "start of converting to bam for $2"
samtools view -u $DIR/bam/$FASTQFILENAME.minimap2.sam | samtools sort -o $DIR/bam/$FASTQFILENAME.minimap2.bam
samtools index $DIR/bam/$FASTQFILENAME.minimap2.bam
echo "end of converting to bam for $2"

echo "start of delitting the sam $DIR/bam/$FASTQFILENAME.bwa.default.sam"
rm $DIR/bam/$FASTQFILENAME.bwa.default.sam
echo "end of deleting of $DIR/bam/$FASTQFILENAME.bwa.default.sam"


echo "start of preparing separating files $DIR/bam/$FASTQFILENAME.minimap2.bam"
samtools view $DIR/bam/$FASTQFILENAME.minimap2.bam | cut -f 1,2,3,4 > $DIR/separating_files/$FASTQFILENAME.minimap2.bam.truncated.txt
echo "end of preping for separation of $DIR/bam/$FASTQFILENAME.minimap2.sam"

## Change to location of picard, or delete if alignment stats are not required
echo "start of collectSummaryMetrics for $DIR/bam/$FASTQFILENAME.minimap2.bam"
java -Xmx85g -jar ~/tools/picard.jar CollectAlignmentSummaryMetrics R=$DIR/$REFERENCEFASTA I=$DIR/bam/$FASTQFILENAME.minimap2.bam \
    O=$DIR/aln_statistics/$FASTQFILENAME.minimap2.SummaryMetrics.log
echo "End of collectSummaryMetrics for $DIR/bam/$FASTQFILENAME.bwa.default.bam"

## delete if alignment statistics are not required
echo "starting getting samtools for bam $2 and fastq ./fastq/$FASTQFILENAME.fastq"
samtools idxstats $DIR/bam/$FASTQFILENAME.minimap2.bam > $DIR/aln_statistics/$FASTQFILENAME.minimap2.idxstats.log
echo "end of getting samtools"


echo "end of script"
