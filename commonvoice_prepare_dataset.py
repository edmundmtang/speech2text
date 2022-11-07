"""Utility script converting commonvoice dataset into wav and json files for training.


"""
import sys
sys.path.append("C:\program files\python310\lib\site-packages") # needed to run in jupyter notebook
sys.path.append("F:\Program Files (x86)\ffmpeg-2022-11-03-git-5ccd4d3060-essentials_build\bin")

import os
import argparse
from tqdm import tqdm
from tqdm import trange
import json
import csv
from time import sleep
from pydub import AudioSegment


def main(args):
    data = []
    directory = args.file_path.rpartition('/')[0]
    percent = args.split_percent
    
    with open(args.file_path, encoding="utf8") as f:
        length = sum(1 for line in f)

    #if args.verbose:
    #    print('Number of audio samples: ', length)

    for i in tqdm(range(100)):
        sleep(0.01)
            
##    with open(args.file_path, newline='', encoding="utf8") as csvfile:
##        reader = csv.DictReader(csvfile, delimiter='\t')
##        index = 1
##        if(args.convert and args.verbose):
##            print(length, "Converting audio samples from mp3 to wav")
##        for row, _ in zip(reader, tqdm(range(length))):
##            file_name = row['path']
##            new_file_name = file_name.rpartition('.')[0] + ".wav"
##            text = row['sentence']
##            data.append({
##                    "key": directory + '/wav/' + new_file_name,
##                    "text": text
##                    })
##            if(args.convert):
##                if args.verbose:
##                    print("converting file " + str(index) + "/" + str(length) + " to wav", end="\r")
##                src = directory + '/clips/' + file_name
##                dst = directory + '/wav/' + new_file_name
##                sound = AudioSegment.from_mp3(src)
##                sound.export(dst, format='wav')
##                index += 1
            

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="""
    Utility script converting commonvoice dataset into wav and json files for training.""")
    parser.add_argument('--file_path', type=str, default=None, required=True,
                        help='path to one of the .tsv files found in cv-corpus')
    parser.add_argument('--split_percent', type=int, default=10, required=False,
                        help='percent of clips to put into test.json instead of train.json')
    parser.add_argument('--convert', default=False, action='store_true',
                        help='tells the script to convert mp3 to wav')
    parser.add_argument('--verbose', action='store_true', help='increase output verbosity')
    args = parser.parse_args()
    main(args)
