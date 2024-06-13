from glob import glob
import os
names = [
        'gsm8k-pathfinder-llama3-8b-new-mcts-8',
         'gsmhard-pathfinder-llama3-8b-new-mcts-8',
         'olympiadbench-pathfinder-llama3-8b-new-mcts-8',
         'GAIC-pathfinder-llama3-8b-new-mcts-8',
         'MATH-pathfinder-llama3-8b-new-mcts-8',
         'AIME-pathfinder-llama3-8b-mcts-2'
         ]

for name in names:
    os.system(f'rm {name}/jsons/*.lock')