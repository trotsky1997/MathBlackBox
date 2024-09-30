from functools import lru_cache
import os
import json
from glob import glob
import random
import re
from tqdm import tqdm
from itertools import groupby
import numpy as np

data_folders = [
    './jsons'
]

pattern = re.compile(r'\-?\d+\.\d+|\-?\d+')

@lru_cache(1024)
def extract_label(text: str) -> str:

    if '\n####' in text:
        text = text.split('\n####')[-1].replace(',','')
    elif 'The answer is' in text:
        text = text.split('The answer is')[-1].replace(',','')
    numbers = pattern.findall(text)
    if not numbers:
        return None
    return numbers[0]

def check(gt,ans):

    gt_label = extract_label(gt)
    ans_label = extract_label(ans)
    # print(gt_label,ans_label)
    if gt_label is None or ans_label is None:
        return False
    if ans_label == gt_label or abs(float(ans_label) - float(gt_label)) < 1e-5:
        return True
    else:
        return False

final_json_list = []

def get_json(query,good,bad,history=[]):
    return {'input':'','instruction':query,'output':[good,bad],'history':history}

def get_node_id(answers,ans):
    return answers.index(ans)

def get_oldest_father(answers,ans,childs):
    possible_fathers = []
    for possible_father in childs:
        if ans in childs[possible_father]:
            possible_fathers.append(possible_father)
    print(len(possible_fathers))
    possible_father_ids = []
    for possible_father in possible_fathers:
        possible_father_ids.append(get_node_id(answers,possible_father))
    return possible_fathers[possible_father_ids.index(min(possible_father_ids))]

def get_fathers(answers,ans,childs):
    possible_fathers = []
    for possible_father in childs:
        if ans in childs[possible_father]:
            possible_fathers.append(possible_father)
    return possible_fathers

def fix_loops(answers,fathers,childs):
    # 如果节点已经在fathers字典中，说明找到了环的起点
    fathers_no_loop = {}
    for node in childs:
        for child in childs[node]:
            if child not in fathers_no_loop:
                fathers_no_loop[child] = [node,]
            elif child in fathers_no_loop:
                fathers_no_loop[child].append(child)

    for ans in answers:
        if ans not in fathers_no_loop:
            fathers_no_loop[ans] = [None,]

    return fathers_no_loop

def collect_paths(answers,gt,fathers,childs):
    gold = []
    for answer in answers:
        if check(gt,answer):
            gold.append(answer)
    paths = []
    for g in gold:
        if g is None:
            continue
        path = [g,]
        while g in fathers and g is not None:
            father = None
            for t in fathers:
                if t in path:
                    continue
                else:
                    father =t
            g = father
            if g is not None:
                path.append(g)
            else:
                break
        paths.append(path)
    return paths

def rereward(paths,answers,gt,fathers,childs,gemma=0.9):
    structue_reward = {}
    for path in paths:
        for i,ix in enumerate(path):
            structue_reward[ix] = gemma**i
    path_list = []
    for path in paths:
        path_list.extend(path)       
    
    gemma2 = 0.5*gemma
    root_reward = min(structue_reward.values())*gemma
    
    def get_reward(ans):
        if ans is None:
            structue_reward[ans] = root_reward
            return structue_reward[ans]

        if ans in path_list:
            return structue_reward[ans]

        if ans in structue_reward:
            return structue_reward[ans]
        if ans in fathers:
            if fathers[ans] is None:
                structue_reward[ans] = root_reward * gemma2
                return structue_reward[ans]
            if fathers[ans] in structue_reward:
                structue_reward[ans] = structue_reward[fathers[ans]] * gemma2
                return structue_reward[ans]
            else:
                structue_reward[ans] = get_reward(fathers[ans]) * gemma2
                return structue_reward[ans]
    for ans in answers:
        get_reward(ans)

    return structue_reward

def get_refined_ans(history_bank,hints_list,answer_list):
    hints_map = {}
    for ans in history_bank:
        if len(history_bank[ans]) > 2:
            hint = history_bank[ans][-3]
            hints_map[hint] = ans
    for hint in hints_list:
        if hint not in hints_map:
            for history in history_bank.values():
                if hint in history:
                    hints_map[hint] = history[history.index(hint)+2]
                    break
    dummys = ["I Don't Know","I can't understand this question.","I can't help with this question.","I don't know how to solve this question.","I don't know the answer to this question.","I don't know the answer to this question, sorry."]
    startpoint = 1
    for dummy in dummys:
        if dummy in answer_list:
            startpoint = answer_list.index(dummy) + 1
    for hint in hints_list:
        if hint not in hints_map:
            hints_map[hint] = answer_list[hints_list.index(hint) + startpoint]
    return hints_map


def collect_refine(paths,hints_reward_imp_bank,hints_map,structure_reward):
    re_hints_reward_imp_bank = {}
    for ans in hints_reward_imp_bank:
        if len(hints_reward_imp_bank[ans]) >= 2:
            re_hints_reward_imp_bank[ans] = []
            for hint,_ in hints_reward_imp_bank[ans]:
                reward0 = structure_reward[ans]
                refined_ans = hints_map[hint]
                reward1 = structure_reward[refined_ans]
                re_hints_reward_imp_bank[ans].append([hint,reward1-reward0])
            re_hints_reward_imp_bank[ans] = sorted(re_hints_reward_imp_bank[ans], key=lambda x: x[1], reverse=True)
            re_hints_reward_imp_bank[ans] = [random.choice(list(g))[0] for k, g in groupby(re_hints_reward_imp_bank[ans], key=lambda x: x[1])]
    return re_hints_reward_imp_bank

def pair_importance_sampling(rewards, actions, nums):
    # Initialize an empty list to store the importance weights
    weights = []
    action_pairs = []
    
    # For each pair of actions
    for i in range(len(actions)):
        for j in range(i+1, len(actions)):
            # Calculate the difference in rewards
            reward_diff = abs(rewards[i] - rewards[j])
            
            # Use the reward difference as the weight for this pair
            weights.append(reward_diff)
            if rewards[i] >= rewards[j]:
                action_pairs.append([actions[i],actions[j]])
            else:
                action_pairs.append([actions[j],actions[i]])
    
    # Normalize the weights so they sum to 1
    weights = [weight / sum(weights) for weight in weights]
    action_pairs_index = list(range(len(action_pairs)))
    
    # Sample from the actions according to the weights
    sampled_actions_index = np.random.choice(action_pairs_index, size=nums, p=weights)
    sampled_actions = [action_pairs[index] for index in sampled_actions_index]
    
    return sampled_actions


def refine_prompt(query,ans):
    q = f'Since we have a weak Answer, could you provide me with a relection or feedback to correct this answer better? Analyze this Answer Strictly and Critic, point out every flaw for every possible imperfect to minus every possible score!\nLet\'s think step by step.'
    return q

for data_folder in data_folders:
    for file in tqdm(glob(data_folder+'/*')):
    # for file in tqdm(glob('/home/bingxing2/ailab/group/ai4phys/math/gsm8k-pathfinder-mistral7B-new-mcts-7/jsons/0913cc66580de7f71567cee17c96479a.json')):
        # print(file)
        data = json.load(open(file,'r'))
        gold = []
        for answer in data['answers_list']:
            if check(data['ground_truth'],answer):
                gold.append(answer)
        data['fathers'] = fix_loops(data['answers_list'],data['fathers'],data['childs'])
        golden_paths = collect_paths(data['answers_list'],data['ground_truth'],data['fathers'],data['childs'])
        structue_reward = rereward(golden_paths,data['answers_list'],data['ground_truth'],data['fathers'],data['childs'])
        hints_map = get_refined_ans(data['history_bank'],data['hints_list'],data['answers_list'])
        re_hints_reward_imp_bank = collect_refine(golden_paths,data['hints_reward_imp_bank'],hints_map,structue_reward)
        dpo_pairs = [] #q,good,bad
        for path in golden_paths:#golden path from right answer to wrong root answers
            if len(path) > 1:
                for i,ix in enumerate(path):
                    # if ix in ["I Don't Know","I can't understand this question.","I can't help with this question.","I don't know how to solve this question.","I don't know the answer to this question.","I don't know the answer to this question, sorry."]:
                    #     path.remove(ix)
                    if ix in gold and i != 0:
                        path.remove(ix)
            if len(path) <= 1:
                golden_paths.remove(path)
        for path in golden_paths:
            if len(path) == 2:
                dpo_pairs.append([data['query'],path[0],path[-1]])
            else:
                pairs = pair_importance_sampling([structue_reward[node] for node in path],path,(len(path)**2)//2)
                pairs = [[data['query'],pair[0],pair[-1]] for pair in pairs]
                dpo_pairs.extend(pairs)
                    # if i < 1:
                    #     continue
                    # else:
                    #     dpo_pairs.append([data['query'],path[i-1],ix])
        # for ans in re_hints_reward_imp_bank:
        #     if ans in ["I Don't Know","I can't understand this question.","I can't help with this question.","I don't know how to solve this question.","I don't know the answer to this question.","I don't know the answer to this question, sorry."]:
        #         continue
        #     if len(re_hints_reward_imp_bank[ans]) >= 2:
        #         for i,hint in enumerate(re_hints_reward_imp_bank[ans]):
        #             if i < 1:
        #                 continue
        #             dpo_pairs.append([refine_prompt(data['query'],ans),re_hints_reward_imp_bank[ans][i-1],hint,[[data['query'],ans],]])
        for dpo_pair in dpo_pairs:
            final_json_list.append(get_json(*dpo_pair))


with open('data_mistral7b_pathfinder_new_mcts_answers_10_percent.json','w') as f:
    random.shuffle(final_json_list)
    print(len(final_json_list))
    json.dump(final_json_list[:len(final_json_list)//100],f)
