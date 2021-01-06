#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import inspect
import argparse

def main():
    chrom_name = args.chrom
    gene_start = int(args.start)
    gene_end = int(args.end)
    gene_name = args.gene
    truncatedFilePath = args.truncatedFile
    sampleNamePrefix = args.sampleName
    print (chrom_name, gene_start, gene_end, gene_name, truncatedFilePath)
    filename = inspect.getframeinfo(inspect.currentframe()).filename
    dir_of_file_path = os.path.dirname(os.path.abspath(filename))
    TruncatedFile = truncatedFilePath

    num_of_reads = 30000000000

    numOfReadsForGene = 0
    numOfReadsForGene_unwanted = 0
    validChrom = [
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "10",
            "11",
            "12",
            "13",
            "14",
            "15",
            "16",
            "17",
            "18",
            "19",
            "20",
            "21",
            "22",
            "X",
            "Y",
            "MT"]
    chr_ID = []
    chr_ID_unwanted = []
    with open(TruncatedFile) as input_file:
        for line in input_file:
            ID, data, chr_num, place = (
            item.strip() for item in line.split())
            ID_validID = ID.split('_')[0]
            if (chr_num == chrom_name) and ((int(place) > (gene_start-20000)) and (int(place) < (gene_end+20000))):
                numOfReadsForGene += 1
                chr_ID.append(ID_validID)
                if numOfReadsForGene > num_of_reads:
                    break
            elif (chr_num in validChrom):
                numOfReadsForGene_unwanted += 1
                chr_ID_unwanted.append(ID_validID)


    print(numOfReadsForGene)
    print("is the number of wanted reads")
    print(numOfReadsForGene_unwanted)
    print("is the number of unwanted reads")
                


    saveFolder = "separating_files/"
    f = open ("%s%s.%s.readIDList.txt" % (saveFolder, sampleNamePrefix, str(gene_name)),"w")
    for ID in chr_ID:
        f.write("%s\n" % ID)
    saveFolder_unwanted = "separating_files/"
    f_unwanted = open ("%s%s.not-%s.readIDList.txt" % (saveFolder_unwanted,sampleNamePrefix, str(gene_name)),"w")
    for ID in chr_ID_unwanted:
        f_unwanted.write("%s\n" % ID)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Get readID for reads located on genes based on gene locations')
    parser.add_argument('--chrom', required=True,
                        help='chrom on whith the gene located')
    parser.add_argument('--start', required=True,
                        help='start position of gene')
    parser.add_argument('--end', required=True,
                        help='end position of gene')
    parser.add_argument('--gene', required=True,
                        help='geneID - Symbol')
    parser.add_argument('--truncatedFile', required=True,
                        help='path to truncated bam file')
    parser.add_argument('--sampleName', required=True,
                        help='sample name added as a prefix')
    args = parser.parse_args()
    main()
