import json
import os
from collections import Counter

screen_names = Counter()

data_path = 'congresstweets/data'
for filename in sorted(os.listdir(data_path)):
    path = os.path.join(data_path,filename)
    with open(path) as f:
        tweets = json.loads(f.read())
        for tweet in tweets:
            try:
                screen_names[tweet['screen_name']]+=1
            except KeyError:
                pass

import pprint
screen_names = dict(screen_names)
pprint.pprint(screen_names)

with open('class_labels.json','w') as f:
    f.write(json.dumps(screen_names))
