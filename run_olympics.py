import os


model = 'meta-llama/Meta-Llama-3-8B-Instruct'

names = [
        'gsm8k-llama3-8b-new-mcts-8',
         'gsmhard-llama3-8b-new-mcts-8',
         'olympiadbench-llama3-8b-new-mcts-8',
         'GAIC-llama3-8b-new-mcts-8',
         'MATH-llama3-8b-new-mcts-8',
         'AIME-llama3-8b-mcts-2'
         ]
# DATA_NAME = 'gsmhard-llama3-8b-new-mcts-8'
# DATA_NAME = 'olympiadbench-llama3-8b-new-mcts-8'
# DATA_NAME = 'GAIC-llama3-8b-new-mcts-8'
# DATA_NAME = 'MATH-llama3-8b-new-mcts-8'
# DATA_NAME = 'AIME-llama3-8b-mcts-2']
for name in names:
    os.system(f'sbatch -p vip_gpu_ailab -A ai4phys --gres=gpu:1 batch.sh {model} {name}')