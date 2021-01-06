#!/bin/csh

## set location of reference genome
## in this example there was a GRCh38 human reference genome in the same folder which is ommited for github size limit
set REFERENCEFASTA=./Homo_sapiens.GRCh38.dna.primary_assembly.fa

foreach f (`ls ./fastq/*.fastq`)
    qsub  ./runMinimap2.sh $REFERENCEFASTA $f

end

