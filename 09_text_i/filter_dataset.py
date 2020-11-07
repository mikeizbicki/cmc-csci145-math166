#!/usr/bin/python3

# process command line arguments
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--input_path', default='congresstweets/data')
parser.add_argument('--output_path', default='congresstweets/data_filtered')
args = parser.parse_args()

# load imports
import json
import os
from collections import Counter
import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    )

# we will filter the input dataset to keep only the 10 senators/representatives
# who are most active on twitter; this is 6 dems and 4 reps;
# list from: https://www.washingtonpost.com/graphics/2019/lifestyle/magazine/amp-stories/twitter/
valid_names = [
        'SenSanders',
        'BernieSanders',

        'SenWarren',
        'ewarren',

        'AOC',
        'aocenespanol',

        'SenBookerOffice',
        'SenBooker',
        'CoryBooker',

        'KamalaHarris',
        'SenKamalaHarris',

        'SpeakerPelosi',
        'TeamPelosi',

        'DrRandPaul',
        'RandPaul',

        'marcorubio',
        'SenRubioPress',

        'SenTedCruz',
        'TeamTedCruz',
        'tedcruz',

        'SenatorRomney',
        'MittRomney',
        ]

# create the output folder
try:
    os.makedirs(args.output_path)
except FileExistsError:
    logging.error('args.output_path='+args.output_path+' already exists')

# loop through the data
screen_names = Counter()
for filename in sorted(os.listdir(args.input_path)):
    logging.info('filename='+filename)

    # compute the valid_tweets in the file
    valid_tweets = []
    path_in = os.path.join(args.input_path,filename)
    with open(path_in,encoding='utf-8') as f_in:
        tweets = json.loads(f_in.read())
        for tweet in tweets:
            try:
                screen_names[tweet['screen_name']]+=1
                if tweet['screen_name'] in valid_names:
                    valid_tweets.append(tweet)
            except KeyError:
                pass

    # save the valid_tweets
    path_out = os.path.join(args.output_path,filename)
    with open(path_out,'xt',encoding='utf-8') as f_out:
        f_out.write(json.dumps(valid_tweets))


# output the class_labels.json file
import pprint
screen_names = dict(screen_names)
screen_names = { name:screen_names[name] for name in valid_names }
pprint.pprint(screen_names)
with open('class_labels.json','w',encoding='utf-8') as f:
    f.write(json.dumps(screen_names))
