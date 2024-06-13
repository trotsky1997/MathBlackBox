import copy
from curses.ascii import isalpha, isdigit
import math
import multiprocessing
import os
import re
import socket
import sys
from datasets import load_dataset
import hashlib
import json
import random
from functools import lru_cache
import numpy as np
from tqdm import tqdm
import time
from retry import retry
import random


# MODEL_NAME = 'meta-llama/Llama-2-7b-chat-hf'
# MODEL_NAME  = 'mistralai/Mistral-7B-Instruct-v0.2'
# MODEL_NAME  = 'meta-llama/Meta-Llama-3-8B-Instruct'
# MODEL_NAME = 'google/gemma-1.1-7b-it'
# MODEL_NAME = 'test-lora'
# MODEL_NAME = '/home/bingxing2/ailab/group/ai4phys/EXPORT/new_mistral_7b_4'
MODEL_NAME = ''


# DATA_NAME = 'meta-math-40k-pathfinder-mistral7B'
# DATA_NAME = 'meta-math-40k-pathfinder-llama2_7B'
# DATA_NAME = 'meta-math-40k-testtime-llama2_7B'
# DATA_NAME = 'gsm8k-rs-llama2_7B'
# DATA_NAME = 'meta-math-40k-testtime-mistral7B'
# DATA_NAME = 'gsm8k-rs-mistral7B'
# DATA_NAME = 'gsm8k-sample-testtime-mistral-dpo-7'
# DATA_NAME = 'gsm8k-testtime-mistral_7B_pathfinder_0'
# DATA_NAME = 'MATH-rs-mistral7B'
# DATA_NAME = 'gsm8k-pathfinder-gemma7b-new-mcts-8'

# DATA_NAME = 'gsmhard-pathfinder-llama3-8b-new-mcts-8'
# DATA_NAME = 'olympiadbench-pathfinder-llama3-8b-new-mcts-8'
# DATA_NAME = 'GAIC-pathfinder-llama3-8b-new-mcts-8'
# DATA_NAME = 'MATH-pathfinder-llama3-8b-new-mcts-8'
# DATA_NAME = 'AIME-pathfinder-llama3-8b-mcts-2'
# DATA_NAME = 'gsm8k-testtime-pathfinder-mistral7B-mcts-2'
# DATA_NAME = 'gsm8k-testtime-pathfinder-pureseq-mistral7B-5'
DATA_NAME = ''

if MODEL_NAME == '':
    MODEL_NAME = sys.argv[1]

if DATA_NAME == '':
    DATA_NAME = sys.argv[2]

def last_boxed_only_string(string):
    idx = string.rfind('\\boxed')
    if idx < 0:
        idx = string.rfind('\\fbox')
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == '{':
            num_left_braces_open += 1
        if string[i] == '}':
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]

    return retval


def remove_boxed(s):
    left = '\\boxed{'
    try:
        assert s[:len(left)] == left
        assert s[-1] == '}'
        return s[len(left):-1]
    except Exception:
        return None


def extract_boxed_answer(pred_str, strip_double_curly_brace=False):
    boxed_str = last_boxed_only_string(pred_str)
    if boxed_str is None:
        return None
    answer = remove_boxed(boxed_str)
    if answer is None:
        return None
    if strip_double_curly_brace:
        match = re.match('^\{(.*)\}$', answer)  # noqa: W605
        if match:
            answer = match.group(1)
    return answer


class Extractor:

    def extract_matching_bracket(cls, target_str: str):
        if not target_str:
            return target_str
        current_nest_level = 1
        for i, ch in enumerate(target_str):
            if ch == '{':
                current_nest_level += 1
            elif ch == '}':
                current_nest_level -= 1
            if current_nest_level == 0:
                break
        return target_str[:i]

    def clean(cls, target_str: str):
        opt = target_str.strip().replace('{{', '{').replace('}}', '}')
        if not opt:
            return opt
        if opt[-1] == '.' or opt[-1] == '。':
            return opt[:-1]
        return opt

    def extract_answer(cls, pred: str, extract_last_num=False):
        if pred.find('The final answer is ') >= 0:
            x = pred[pred.find('The final answer is ') +
                     len('The final answer is '):]
            x = x[1:x.find('$.')]
            # print(x)
            return cls.clean(x)
        if pred.find('\n\nQuestion:') >= 0:
            pred = pred.split('\n\nQuestion:')[0]
            if pred.find('The answer is'):
                pred = pred[pred.find('The answer is') + len('The answer is'):]
                return cls.clean(pred)
        if pred.find('# Answer') >= 0:
            return cls.clean(pred[pred.find('# Answer') + len('# Answer'):])
        if pred.find('The answer is:') >= 0:
            return cls.clean(pred[pred.find('The answer is:') +
                                  len('The answer is:'):])
        if pred.find('####') >= 0:
            return cls.clean(pred[pred.find('####') + 4:])
        left = '\\boxed{'
        if pred.find(left) >= 0:
            pred = pred[pred.find(left) + len(left):]
            return cls.clean(cls.extract_matching_bracket(pred))

        if extract_last_num:
            nums = []
            opt = ''

            def contain_digit(opt):
                for ch in opt:
                    if ch.isdigit():
                        return True
                return False

            for ch in pred:
                if ch.isdigit() or ch in ' ,.':
                    opt = opt + ch
                else:
                    if contain_digit(opt):
                        nums.append(opt)
                    opt = ''
            if contain_digit(opt):
                return cls.clean(opt)
            if nums:
                return cls.clean(nums[-1])
        return None


def fix_fracs(string):
    substrs = string.split('\\frac')
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += '\\frac'
            if substr[0] == '{':
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except AssertionError:
                    return string
                a = substr[0]
                b = substr[1]
                if b != '{':
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += '{' + a + '}{' + b + '}' + post_substr
                    else:
                        new_str += '{' + a + '}{' + b + '}'
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += '{' + a + '}' + b + post_substr
                    else:
                        new_str += '{' + a + '}' + b
    string = new_str
    return string


def fix_a_slash_b(string):
    if len(string.split('/')) != 2:
        return string
    a = string.split('/')[0]
    b = string.split('/')[1]
    try:
        a = int(a)
        b = int(b)
        assert string == '{}/{}'.format(a, b)
        new_string = '\\frac{' + str(a) + '}{' + str(b) + '}'
        return new_string
    except AssertionError:
        return string


def remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set)
    if '\\text{ ' in string:
        splits = string.split('\\text{ ')
        assert len(splits) == 2
        return splits[0]
    else:
        return string


def fix_sqrt(string):
    if '\\sqrt' not in string:
        return string
    splits = string.split('\\sqrt')
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != '{':
            a = split[0]
            new_substr = '\\sqrt{' + a + '}' + split[1:]
        else:
            new_substr = '\\sqrt' + split
        new_string += new_substr
    return new_string


def strip_string(string):
    # linebreaks
    string = string.replace('\n', '')

    # remove inverse spaces
    string = string.replace('\\!', '')

    # replace \\ with \
    string = string.replace('\\\\', '\\')

    # replace tfrac and dfrac with frac
    string = string.replace('tfrac', 'frac')
    string = string.replace('dfrac', 'frac')

    # remove \left and \right
    string = string.replace('\\left', '')
    string = string.replace('\\right', '')

    # Remove circ (degrees)
    string = string.replace('^{\\circ}', '')
    string = string.replace('^\\circ', '')

    # remove dollar signs
    string = string.replace('\\$', '')

    # remove units (on the right)
    string = remove_right_units(string)

    # remove percentage
    string = string.replace('\\%', '')
    string = string.replace('\%', '')  # noqa: W605

    string = string.replace(' .', ' 0.')
    string = string.replace('{.', '{0.')
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == '.':
        string = '0' + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split('=')) == 2:
        if len(string.split('=')[0]) <= 2:
            string = string.split('=')[1]

    # fix sqrt3 --> sqrt{3}
    string = fix_sqrt(string)

    # remove spaces
    string = string.replace(' ', '')

    string = fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == '0.5':
        string = '\\frac{1}{2}'

    string = fix_a_slash_b(string)
    string = string.replace('x \\in', '').strip()  # noqa: W605

    # a_b == a, a_{b} == a_b for bit conversion
    if string.find('_') >= 0:
        p = string.split('_')
        p[1] = p[1].replace('{', '').replace('}', '')
        string = '_'.join(p)

    # 10800 == 10,800; we only deal with single number
    if string.strip().find(' ') == -1 and string.find('(') == -1:
        string = string.replace(',', '')

    return string


def is_equiv(str1, str2, verbose=False):
    if str1 is None and str2 is None:
        # print("WARNING: Both None")
        return False
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = strip_string(str1)
        ss2 = strip_string(str2)
        return ss1 == ss2
    except Exception:
        return str1 == str2


if not os.path.exists(DATA_NAME):
    os.mkdir(DATA_NAME)
if not os.path.exists(f'{DATA_NAME}/jsons'):
    os.mkdir(f'{DATA_NAME}/jsons')

if 'testtime' in DATA_NAME:
    if 'gsm8k' in DATA_NAME:
        if 'sample' in DATA_NAME:
            dataset = load_dataset("gsm8k",'main',split='test')
            # dataset = dataset.shuffle()
            dataset = dataset.select(range(130))
        else:
            dataset = load_dataset("gsm8k",'main',split='test')
    elif 'MATH' in DATA_NAME:
        dataset = load_dataset("lighteval/MATH",'all',split='test')
else:
    if 'gsmhard' in DATA_NAME:
        dataset = load_dataset("reasoning-machines/gsm-hard",split='train')
    elif 'gsm8k' in DATA_NAME:
        if not 'mcts' in DATA_NAME:
            dataset = load_dataset("gsm8k",'main',split='train')
        else:
            dataset = load_dataset("gsm8k",'main',split='test')
    elif 'level5' in DATA_NAME:
        dataset = load_dataset("lighteval/MATH",'all',split='test',trust_remote_code=True)
        dataset = dataset.filter(lambda example: example["level"].endswith("5"))
    elif 'MATH' in DATA_NAME and not'level5' in DATA_NAME:
        dataset = load_dataset("lighteval/MATH",'all',split='test',trust_remote_code=True)
    elif 'AIME' in DATA_NAME:
        dataset = load_dataset("qq8933/AIME_1983_2024",split='train')
    elif 'olympiadbench' in DATA_NAME:
        dataset = load_dataset("lmms-lab/OlympiadBench",split='test_en')
        dataset = dataset.filter(lambda example:len(example["images"]) == 0 and example['final_answer'] is not None and len(example['final_answer']) == 1)
    elif 'meta-math' in DATA_NAME:
        dataset = load_dataset("meta-math/MetaMathQA-40K",split='train')
    elif 'GAIC' in DATA_NAME:
        dataset = load_dataset("qq8933/AGI_Odyssey_MATH_GAIC_2024")
    elif 'mathinstruct' in DATA_NAME:
        dataset = load_dataset('TIGER-Lab/MathInstruct',split='train')
    else:
        dataset = load_dataset('json',data_files=f'/home/bingxing2/ailab/group/ai4phys/math/data_mistral_var_sft.json')

dataset.shuffle()
from openai import OpenAI

# generation_lock = multiprocessing.Lock()






# def generate(prompt,):
#     with generation_lock:
#         ret = generate_in(prompt,)
#     return ret

# client = OpenAI(
#     base_url="http://10.140.24.132:10087/v1",
#     api_key="token-abc123",
# )

clients = []
times = time.time()
# def get_clients():
#     global clients
#     lines = open('/mnt/petrelfs/zhangdi1/reasoningpath/math/server.csv','r').readlines()
#     for line in lines:
#         if len(line) < 3:
#             continue
#         node,port,model = line.split(',')
#         ip = '.'.join(node.split('-')[-4:])
#         client = OpenAI(
#         base_url=f"http://{ip}:{port}/v1",
#         api_key="token-abc123",
#         )
#         try:
#             client.chat.completions.create(
#             model=MODEL_NAME,
#             messages=[
#                 {"role": "user", "content": 'hi'}#+'\nBe concisely and clearly in no more than 50 words.'
#             ],
#             # max_tokens=min(len(prompt)+128,8000),
#             temperature=0.95,
#             timeout=10
#             )
#             print(len(clients)+1)
#             clients.append(client)
#         except:
#             pass

from concurrent.futures import ThreadPoolExecutor

def create_client(line):
    global clients
    if len(line) < 3:
        return
    node,port,model = line.split(',')
    ip = socket.gethostbyname(node)
    print(ip)
    client = OpenAI(
    base_url=f"http://{ip}:{port}/v1",
    api_key="token-abc123",
    )
    try:
        client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "user", "content": 'hi'}#+'\nBe concisely and clearly in no more than 50 words.'
        ],
        # max_tokens=min(len(prompt)+128,8000),
        temperature=0.95,#0.5 if 'testtime' in DATA_NAME else random.uniform(0,1)
        timeout=15
        )
        print(len(clients)+1)
        clients.append(client)
    except:
        pass

def get_clients():
    global clients
    lines = open('./server.csv','r').readlines()
    with ThreadPoolExecutor() as executor:
        executor.map(create_client, lines)




def get_client():
    global clients,times
    # if time.time() - times > 1800:
    #     clients = []
    #     get_clients()
    #     times = time.time()
    return random.choice(clients)


@retry()
def generate(prompt,history=[],timeout = 150,truncate=True):
    if 'testtime' in DATA_NAME:
        timeout=150
    print('awaiting response...')
    time0 = time.time()
    history_ = [{"role": "user" if i %2 ==0 else 'assistant', "content": h} for i,h in enumerate(history)]
    if truncate:
        history_ = history_[-2:]
    completion = get_client().chat.completions.create(
    model=MODEL_NAME,
    messages=history_+[
                # dict(role='user', content="Question: Angelo and Melanie want to plan how many hours over the next week they should study together for their test next week. They have 2 chapters of their textbook to study and 4 worksheets to memorize. They figure out that they should dedicate 3 hours to each chapter of their textbook and 1.5 hours for each worksheet. If they plan to study no more than 4 hours each day, how many days should they plan to study total over the next week if they take a 10-minute break every hour, include 3 10-minute snack breaks each day, and 30 minutes for lunch each day?\nLet's think step by step\nAnswer:"),
                # dict(role='assistant', content="Angelo and Melanie think they should dedicate 3 hours to each of the 2 chapters, 3 hours x 2 chapters = 6 hours total.\nFor the worksheets they plan to dedicate 1.5 hours for each worksheet, 1.5 hours x 4 worksheets = 6 hours total.\nAngelo and Melanie need to start with planning 12 hours to study, at 4 hours a day, 12 / 4 = 3 days.\nHowever, they need to include time for breaks and lunch. Every hour they want to include a 10-minute break, so 12 total hours x 10 minutes = 120 extra minutes for breaks.\nThey also want to include 3 10-minute snack breaks, 3 x 10 minutes = 30 minutes.\nAnd they want to include 30 minutes for lunch each day, so 120 minutes for breaks + 30 minutes for snack breaks + 30 minutes for lunch = 180 minutes, or 180 / 60 minutes per hour = 3 extra hours.\nSo Angelo and Melanie want to plan 12 hours to study + 3 hours of breaks = 15 hours total.\nThey want to study no more than 4 hours each day, 15 hours / 4 hours each day = 3.75\nThey will need to plan to study 4 days to allow for all the time they need.\nThe answer is 4\n"),
                # dict(role='user', content="Question: Mark's basketball team scores 25 2 pointers, 8 3 pointers and 10 free throws.  Their opponents score double the 2 pointers but half the 3 pointers and free throws.  What's the total number of points scored by both teams added together?\nLet's think step by step\nAnswer:"),
                # dict(role='assistant', content="Mark's team scores 25 2 pointers, meaning they scored 25*2= 50 points in 2 pointers.\nHis team also scores 6 3 pointers, meaning they scored 8*3= 24 points in 3 pointers\nThey scored 10 free throws, and free throws count as one point so they scored 10*1=10 points in free throws.\nAll together his team scored 50+24+10= 84 points\nMark's opponents scored double his team's number of 2 pointers, meaning they scored 50*2=100 points in 2 pointers.\nHis opponents scored half his team's number of 3 pointers, meaning they scored 24/2= 12 points in 3 pointers.\nThey also scored half Mark's team's points in free throws, meaning they scored 10/2=5 points in free throws.\nAll together Mark's opponents scored 100+12+5=117 points\nThe total score for the game is both team's scores added together, so it is 84+117=201 points\nThe answer is 201\n"),
                # dict(role='user', content="Question: Bella has two times as many marbles as frisbees. She also has 20 more frisbees than deck cards. If she buys 2/5 times more of each item, what would be the total number of the items she will have if she currently has 60 marbles?\nLet's think step by step\nAnswer:"),
                # dict(role='assistant', content="When Bella buys 2/5 times more marbles, she'll have increased the number of marbles by 2/5*60 = 24\nThe total number of marbles she'll have is 60+24 = 84\nIf Bella currently has 60 marbles, and she has two times as many marbles as frisbees, she has 60/2 = 30 frisbees.\nIf Bella buys 2/5 times more frisbees, she'll have 2/5*30 = 12 more frisbees.\nThe total number of frisbees she'll have will increase to 30+12 = 42\nBella also has 20 more frisbees than deck cards, meaning she has 30-20 = 10 deck cards\nIf she buys 2/5 times more deck cards, she'll have 2/5*10 = 4 more deck cards.\nThe total number of deck cards she'll have is 10+4 = 14\nTogether, Bella will have a total of 14+42+84 = 140 items\nThe answer is 140\n"),
                # dict(role='user', content="Question: A group of 4 fruit baskets contains 9 apples, 15 oranges, and 14 bananas in the first three baskets and 2 less of each fruit in the fourth basket. How many fruits are there?\nLet's think step by step\nAnswer:"),
                # dict(role='assistant', content="For the first three baskets, the number of apples and oranges in one basket is 9+15=24\nIn total, together with bananas, the number of fruits in one basket is 24+14=38 for the first three baskets.\nSince there are three baskets each having 38 fruits, there are 3*38=114 fruits in the first three baskets.\nThe number of apples in the fourth basket is 9-2=7\nThere are also 15-2=13 oranges in the fourth basket\nThe combined number of oranges and apples in the fourth basket is 13+7=20\nThe fourth basket also contains 14-2=12 bananas.\nIn total, the fourth basket has 20+12=32 fruits.\nThe four baskets together have 32+114=146 fruits.\nThe answer is 146\n"),
        {"role": "user", "content": prompt}#
    ],
    # max_tokens=min(len(prompt)+128,8000),
    temperature=0.95,#0.5 if 'testtime' in DATA_NAME else random.uniform(0,1),
    timeout = timeout
    )
    print(f'response received! time taken: {time.time()-time0} seconds.')
    return completion.choices[0].message.content,list(history)+[prompt,completion.choices[0].message.content]

@retry()
def cal_reward(question,ans):
    query = f'Question: {question}\nAnswer:{ans}\nAnalyze this Answer Strictly and Critic, point out every flaw for ervery possible imperfect to minus every possible score! You need to be very harsh and mean in calculating grades, and never give full marks to ensure that the marks are authoritative. \nOutput a score between [-100,+100], ig. from -100 to +100. \nResponse format:\n[Analyst]...[Score]...'
    ret = generate(query)
    score = ret[0].split('Score')[-1]
    scores = pattern.findall(score)
    if not scores:
        raise Exception('no')
    else:
        ret = float(scores[-1])
        # if abs(ret - 100.0) < 1e-5:
        #     ret = 50.0
        if ret >= 95:
            ret = 50
        # elif ret <= -100:
        #     ret = -50
        return ret 

@retry()
def get_weak_answer(question,new_len=0,ans_format=''):
    query = f'Question: {question}\nThe response should begin with [reasoning process]...[Verification]... and end with {ans_format}\nLet\'s think step by step.'
    return generate(query,timeout=90)

def get_weak_hints(question,weak_answer,ground_truth_label=None,new_len=0,history=[],alreadygood=False,ans_format=''):
    query = f'Question: {question}\nSince we have a weak Answer, could you provide me with a relection or feedback to correct this answer better? Analyze this Answer Strictly and Critic, point out every flaw for ervery possible imperfect to minus every possible score!\nLet\'s think step by step.'
    return generate(query,history)

def get_better_answer(question,weak_answer,hint,new_len=0,history=[],ans_format=''):
    query = f'Question: {question}\nPlease refine the your answer according to your Reflection or Feedback. The response should begin with [reasoning process]...[Verification]... and end with end with {ans_format}\nLet\'s think step by step.'
    return generate(query,history)

def get_gt_hints(question,ground_truth,new_len=0):
    query = f"Question: {question}\nGround Truth:{ground_truth}\nAccording to ground truth answer we have, Could you descript the thought process of ground truth answer, please don’t give me the answer, just the thought process?"
    return generate(query)


datas = []
pattern = re.compile(r'\-?\d+\.\d+|\-?\d+')
extractor_0 = Extractor()
@lru_cache(1024)
def extract_label(text: str,type='') -> str:
    if 'gsm' not in DATA_NAME and type != 'digit':
        if '####' in text:
            text = text.split('####')[-1]
        elif 'The answer is' in text:
            text = text.split('The answer is')[-1]
            if '####' in text:
                text = text.split('####')[-1]
        if 'box' in text:
            return extract_boxed_answer(text)
        else:
            return text
    if '\n####' in text:
        text = text.split('\n####')[-1].replace(',','')
    elif 'The answer is' in text:
        text = text.split('The answer is')[-1].replace(',','')
    numbers = pattern.findall(text)
    if not numbers:
        return None
    if '\n####' in text or 'The answer is' in text:
        return numbers[0]
    else :
        return numbers[-1]

@lru_cache(1024)
def check(gt,ans):
    gt_label = extract_label(gt)
    if gt_label.isdigit():
        type = 'digit'
    elif gt_label.isupper() and gt_label.isalpha():
        type = 'option'
    elif gt_label.lower() in ['yes','no']:
        gt_label = gt_label.lower()
        type = 'yesorno'
    else :
        type = 'formula'
    ans_label = extract_label(ans,type)
    if ans_label:
        if type == 'option':
            ans_label = ans_label.strip()[0]
        elif type == 'yesorno':
            ans_label = ans_label.lower()
        elif type == 'formula':
            ans_label = ans_label.replace('$','')
    print(gt_label,ans_label)
    if 'gsm' not in DATA_NAME and type != 'digit':
        return is_equiv(gt_label,ans_label)
    print(gt_label,ans_label)
    if gt_label is None or ans_label is None:
        return False
    if ans_label == gt_label or abs(float(ans_label) - float(gt_label)) < 1e-5:
        return True
    else:
        return False

def hamming_distance(str1, str2):
    if len(str1) != len(str2):
        raise ValueError("Strings must be of the same length")
    return sum(el1 != el2 for el1, el2 in zip(str1[::-1], str2[::-1]))

def simple_reward(gt,ans):
    gt_f = format(float(extract_label(gt)),'.5f')
    ans_f = format(float(extract_label(ans)),'.5f')
    return -hamming_distance(gt_f,ans_f)

def sort_answers_and_rewards(answers, rewards):
    # Zip answers and rewards together
    answer_reward_pairs = zip(answers, rewards)
    
    # Sort pairs by rewards
    sorted_pairs = sorted(answer_reward_pairs, key=lambda x: x[1], reverse=True)
    
    # Extract sorted answers and rewards
    sorted_answers = [pair[0] for pair in sorted_pairs]
    sorted_rewards = [pair[1] for pair in sorted_pairs]
    
    return sorted_answers, sorted_rewards

def filter_mature_node(childs, to_explore, to_explore_reward,max_expand=3):
    filterd_to_explore = []
    avg_reward = {node: (min(to_explore_reward[node]) + np.mean(to_explore_reward[node])) / 2 for node in to_explore}

    for node in to_explore:
        if len(childs.get(node,[])) < max_expand or max([avg_reward.get(child,-999) for child in childs.get(node,[])]) < avg_reward.get(node,-999):
            filterd_to_explore.append(node)
    
    return filterd_to_explore


def get_best_explore_from_ucb(to_explore, ucb_bank):
    # 初始化最佳节点和最高UCB值
    best_node = None
    highest_ucb = float('-inf')
    
    # 遍历所有待探索的节点
    for node in to_explore:
        ucb_value = ucb_bank.get(node, float('-inf'))
        if ucb_value > highest_ucb:
            highest_ucb = ucb_value
            best_node = node
            
    return best_node


def compute_ucb(r_c, N_n, N_c, C):
    return r_c + C * math.sqrt(math.log(N_n + 1) / (N_c + 1e-5))

def update_ucb(fathers, childs, to_explore, to_explore_reward, ucb_bank, C=1.4,gamma=0.85):
    # 计算所有节点的访问次数
    visit_count = {node: len(to_explore_reward[node]) for node in to_explore}

    # 计算所有节点的平均奖励
    # avg_reward = {node: sum(to_explore_reward[node]) / len(to_explore_reward[node]) for node in to_explore}
    avg_reward = {node: (min(to_explore_reward[node]) + np.mean(to_explore_reward[node])) / 2 for node in to_explore}

    # 获取所有叶子节点
    leaves = set(to_explore) - set(fathers.values())
    
    # 更新所有叶子节点的UCB值
    for leaf in leaves:
        # ucb_bank[leaf] = avg_reward[leaf]
        ucb_bank[leaf] = compute_ucb(avg_reward[leaf],len(to_explore_reward.get(fathers.get(leaf,None),[])),len(to_explore_reward.get(leaf,[])),C)
    
    # 从叶子节点向上更新父节点的UCB值
    nodes_to_update = list(leaves)
    while nodes_to_update:
        new_nodes_to_update = set()
        for node in nodes_to_update:
            father = fathers.get(node)
            if father is not None:
                if father not in ucb_bank:
                    new_nodes_to_update.add(father)
                if father in ucb_bank:
                    # 计算父节点的UCB值
                    ucb_values = []
                    child_reward = []
                    for child in childs[father]:
                        ucb_values.append(ucb_bank[child])
                        child_reward.append(avg_reward[child])
                    father_reward = (avg_reward[father] + max(child_reward))/2
                    ucb_bank[father] = compute_ucb(father_reward,len(to_explore_reward.get(fathers.get(father,None),[])),len(to_explore_reward.get(father,[])),C)
        nodes_to_update = list(new_nodes_to_update)

def step(query,weak_answer,ground_truth_label=None,history=[],alreadygood=False,ans_format=''):
    hints,history = get_weak_hints(query,weak_answer,ground_truth_label=ground_truth_label,history=history,alreadygood=alreadygood,ans_format=ans_format)
    answer,history = get_better_answer(query,weak_answer,hints,history=history,ans_format=ans_format)
    return hints,answer,history

def main_loop(query,ground_truth,max_iter=16,ans_format=''):
    to_explore = []
    to_explore_reward = {}
    history_bank = {}
    hints_bank = {}
    ucb_bank = {}
    fathers = {}
    childs = {}
    def sampling_reward(answer):
        if answer not in to_explore_reward:
            to_explore_reward[answer] = []
        reward = cal_reward(query,answer)
        # if check(ground_truth,answer):
        #     reward += 100
        to_explore_reward[answer].append(reward)

    def add_to_hints_bank(hints,weak_answer):
        if weak_answer not in hints_bank:
            hints_bank[weak_answer] = []
        hints_bank[weak_answer].append(hints)

    def add_to_childs(father,child):
        if father not in childs:
            childs[father] = []
        childs[father].append(child)

    hints_reward_imp_bank = {}
    def add_to_hints_reward_imp_bank(hints,weak_answer,reward,answer):
        if weak_answer not in hints_reward_imp_bank:
            hints_reward_imp_bank[weak_answer] = []
        hints_reward_imp_bank[weak_answer].append((hints,reward,answer))
    ground_truth_label = extract_label(ground_truth)
    ###get weak answer###
    weak_answer,history = get_weak_answer(query,ans_format=ans_format)
    history_bank[weak_answer] = tuple(history)
    answers_list = [weak_answer,]
    to_explore = [weak_answer,]
    childs[weak_answer] = []
    fathers[weak_answer] = None
    # to_explore_reward = [cal_reward(query,weak_answer),]
    sampling_reward(weak_answer)
    ##add total-bad answer###
    # if check(ground_truth,weak_answer):
    #     return 
    if True:#not check(ground_truth,weak_answer):
        total_bad = random.choice(["I Don't Know","I can't understand this question.","I can't help with this question.","I don't know how to solve this question.","I don't know the answer to this question.","I don't know the answer to this question, sorry."])
        total_bad_history = copy.deepcopy(history)
        total_bad_history[-1] = total_bad
        history_bank[total_bad] = tuple(total_bad_history)
        answers_list += [total_bad,]
        to_explore += [total_bad,]
        childs[total_bad] = []
        fathers[total_bad] = None
        # to_explore_reward = [cal_reward(query,weak_answer),]
        sampling_reward(total_bad)
    hints_list = []
    if check(ground_truth,weak_answer) :#and 'testtime' in DATA_NAME
        return hints_list,answers_list,to_explore,to_explore_reward,hints_bank,history_bank,hints_reward_imp_bank,fathers,childs,ucb_bank
    patient = 0 if 'testtime' not in DATA_NAME else 0
    alpha = 0.45
    update_ucb(fathers=fathers,childs=childs,to_explore=to_explore,to_explore_reward=to_explore_reward,ucb_bank=ucb_bank)
    for i in range(max_iter):
        print('iteration:',i)
        filterd_to_explore = filter_mature_node(childs, to_explore, to_explore_reward)
        weak_answer = get_best_explore_from_ucb(filterd_to_explore, ucb_bank)
        sampling_reward(weak_answer)
        hints,answer,history = step(query,weak_answer,history=history_bank[weak_answer],ans_format=ans_format)
        add_to_hints_bank(hints,weak_answer)
        history_bank[answer] = tuple(history)
        to_explore.append(answer)
        sampling_reward(answer)
        fathers[answer] = weak_answer
        childs[answer] = []
        add_to_childs(weak_answer,answer)
        answers_list.append(answer)
        hints_list.append(hints)
        if check(ground_truth,answer) and 'testtime' in DATA_NAME:
            return hints_list,answers_list,to_explore,to_explore_reward,hints_bank,history_bank,hints_reward_imp_bank,fathers,childs,ucb_bank
        elif check(ground_truth,answer) and 'testtime' not in DATA_NAME:
            if patient <= 0:
                return hints_list,answers_list,to_explore,to_explore_reward,hints_bank,history_bank,hints_reward_imp_bank,fathers,childs,ucb_bank
            patient -= 1
        update_ucb(fathers=fathers,childs=childs,to_explore=to_explore,to_explore_reward=to_explore_reward,ucb_bank=ucb_bank)
        add_to_hints_reward_imp_bank(hints,weak_answer,min(to_explore_reward.get(answer)) - min(to_explore_reward.get(weak_answer)),answer)#ucb_bank[answer] - ucb_bank[weak_answer]
    return hints_list,answers_list,to_explore,to_explore_reward,hints_bank,history_bank,hints_reward_imp_bank,fathers,childs,ucb_bank

def tryfunc(example):
    try:
        if os.path.exists(f'{DATA_NAME}/jsons/{hashlib.md5(str(example).encode()).hexdigest()}.json.lock'):
            return
        else:
            os.system(f'touch {DATA_NAME}/jsons/{hashlib.md5(str(example).encode()).hexdigest()}.json.lock')
        func(example)
        if os.path.exists(f'{DATA_NAME}/jsons/{hashlib.md5(str(example).encode()).hexdigest()}.json.lock'):
            os.system(f'rm {DATA_NAME}/jsons/{hashlib.md5(str(example).encode()).hexdigest()}.json.lock')
    except:
        print(example)
        pass
    # for example in tqdm(dataset['train']):

def func(example):
    if os.path.exists(f'{DATA_NAME}/jsons/{hashlib.md5(str(example).encode()).hexdigest()}.json'):
        # return json.load(open(f'{DATA_NAME}/jsons/{hashlib.md5(str(example).encode()).hexdigest()}'))
        return {}
    if 'instruction' in example and 'output' in example:
        query = example['instruction'] + '\n' + example['input']
        ground_truth = example['output']
    elif 'context' in example and 'question' in example:
        if example['context']:
            query = example['context'] + '\n' + example['question']
        else:
            query = example['question']
        ground_truth = example['final_answer'][0].replace('$','')
    elif 'GAIC' in DATA_NAME :
        query = example['problem']
        ground_truth = example['answer']
    else:
        if 'query' in example:
            query = example['query']
        elif 'problem' in example:
            query = example['problem']
        elif 'input' in example:
            query = example['input']
        elif 'Question' in example:
            query = example['Question']
        else:
            query = example['question']
        if 'response' in example:
            ground_truth = example['response']
        elif 'solution' in example:
            ground_truth = example['solution']
        elif 'target' in example:
            ground_truth = str(example['target'])
        elif 'Answer' in example:
            ground_truth = example['Answer']
        else:
            ground_truth = example['answer']

    if 'gsm' in DATA_NAME:
        ans_format = r'"[Final Answer] The answer is [answer] \n#### [answer]"'
    else:
        if extract_label(ground_truth).isdigit():
            ans_format = r'"[Final Answer] The answer is [number] \n#### [number]"'
        elif extract_label(ground_truth).isalpha() and extract_label(ground_truth).isupper():
            ans_format = r'"[Final Answer] The answer is \\boxed{[option]} \n#### [option]"'
        elif extract_label(ground_truth).lower() in ['yes','no']:
            ans_format = r'"[Final Answer] The answer is \\boxed{[Yes or No]} \n#### [Yes or No]"'
        else:
            ans_format = r'"[Final Answer] The answer is \\boxed{[answer formula]} \n#### [answer formula]"'
        
    # new_len = len(ground_truth)
    hints_prompt = f'Question: {query}\nCould you provide me with the thought process to solve this problem, but please don’t give me the answer or calculation, just the thought process?'
    max_iter = 16
    if 'meta-math' in DATA_NAME:
        max_iter = 8
    if 'testtime' in DATA_NAME:
        max_iter = 2
    hints_list,answers_list,to_explore,to_explore_reward,hints_bank,history_bank,hints_reward_imp_bank,fathers,childs,ucb_bank = main_loop(query,ground_truth,max_iter=max_iter,ans_format=ans_format)
    if len(answers_list) <= 1 and 'rs' in DATA_NAME:
        return 
    else:
        if not 'testtime' in DATA_NAME:
            # gt_hints = get_gt_hints(query,ground_truth)
            gt_hints = ''
            pass
        else:
            gt_hints = ''
        data = {
            'query':query,
            'ground_truth':ground_truth,
            'hints_list':hints_list,
            'answers_list':answers_list,
            'ground_truth_hints':gt_hints,
            'hints_prompt':hints_prompt,
            'to_explore':to_explore,
            'to_explore_reward':to_explore_reward,
            'hints_bank':hints_bank,
            'history_bank':history_bank,
            'hints_reward_imp_bank':hints_reward_imp_bank,
            'fathers':fathers,
            'childs':childs,
            'ucb_bank':ucb_bank,
        }
        if 'rs' in DATA_NAME and not check(ground_truth,answers_list[-1]):
            return

        with open(f'{DATA_NAME}/jsons/{hashlib.md5(str(example).encode()).hexdigest()}.json','w+') as f:
            json.dump(data,f,indent=4,ensure_ascii=False)
        
        return data

if __name__ == '__main__':
    get_clients()
    # while True:
    #     try:
    # datas = dataset.map(func,num_proc=len(clients)*8)
    datas = dataset.map(func,num_proc=32)
        # except :
        #     continue
        # break

    # datas.save_to_disk('meta-math-40k-weak-better-mistral7B-data')