# Proccessing Fast5 files for deep learning
## introduction
This is a short guide to proccess raw fast5 (not basecalled) files to csv files fitting for deep learning training.
In addition, this worflow separates signals into two categories for classification: mitochondrial signals, and non-mitochondrial signals.
To change the classification problem of the deep learning models (to seperate signals into "signals from interesting genes" vs "non-interesting signals")
simply change the separation parameters in step 5.

## Requirments
Python (in addition to the requirments in the main folder):
* h5py

Linux:
* Albacore basecaller(by ONT) - Basecalling (translating raw nanopore signal to nucleotide sequence)
* Poretools - Extract fastq files from basecalled fast5 files
* Minimap2 - Align long reads to a reference genome
* Samtools - SAM/BAM manipulation
* Picard (optional) - create alingment statistics 

The scripts were written for a computational cluster environment. In order to run on a machine wethout a job scheduler simply remove `qsub` from scripts. 

## Workflow
1. Make sure all fast5 files are located in a `fast5` folder and then inside a folder with the name of the flowcell, this name should be updated inside `runAlbacore.sh` and `runPoretools.sh`.

Run `./runAlbacore.sh`

Remember to check the kit version and flowcell name in the script

2. After albacore finished run `./runPoretools.sh.`
Remember to change the flowcell name also

3. After poretools finished extracting fast5 run minimap2 with './runMinimap2Multy.sh'
After this finished we have two statistics files and the bam and sam.


Continue if want to separate Fast5 files into two classes and extract the signals from them

4. Add a folder with links to all fast5 files from the experiment with:
```
mkdir linksToAllFast5
cd linksToAllFast5
find ./fast5/20180724_0920_Hek_run1_cont/fast5 -name "*.fast5" -print0 | xargs -0 cp -s -r --target-directory=.
```
5. Create an experiment summary file with:
```
cat basecalls/FAF13387-Hek_run1_cont/*sequencing_summary* >> separating_files/concat_seq_summary.txt
```
6. Create the files containing the IDs of wanted reads from the experiment with:
```
python ../creatReadIDfiles_genes.py --chrom MT --start 0 --end 99999999999 --gene Mito \
--truncatedFile ./separating_files/FAF13387-Hek_run1.minimap2.bam.truncated.txt --sampleName FAF13387-Hek_run1.minimap2
```
7. Change line 33 in the `./runGetLinksAndDownloadBasedOnReadIDListFiles.sh` file to the folder where all the links are
https://github.com/nshomron/Nanopore_Deep_Learning/blob/2f2546e9d8571a5d90e602859e4ab0ea419c2d90/Preparing_fast5_for_analysis/runGetLinksAndDownloadBasedOnReadIDListFiles.sh#L33

8. Run `./runGetLinksAndDownloadBasedOnReadIDListFilesMulty.sh`

9. Check the CSV files in each folder in `separating_files/` check the number of files
