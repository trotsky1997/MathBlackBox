from functools import lru_cache
from glob import glob
import json
import re
import sys

data_list = glob(f"{sys.argv[1]}/jsons/*")

pattern = re.compile(r'\-?\d+\.\d+|\-?\d+')

right = 0
all_fail = 0
truncate = 8

print(len(data_list))

from run_with_earlystopping import check

retry_times = []
retry_times_no_one = []

def checklist(gt,anss):
    for ans in anss:
        if check(gt,ans):
            return True
    return False

for file in data_list:
    with open(file, 'r') as f:
        data = json.load(f)
        retry_time = len(data['answers_list'])
        retry_times.append(retry_time)
        answers = []
        for i in data['answers_list']:
            if i in ["I Don't Know","I can't understand this question.","I can't help with this question.","I don't know how to solve this question.","I don't know the answer to this question.","I don't know the answer to this question, sorry."]:
                pass
            else:
                answers.append(i)
        if checklist(data['ground_truth'],answers):
            right += 1
        else:
            all_fail += 1


import numpy as np


all_len = len(data_list)
print('Acc rate',right,all_len,right/all_len)
