import os


# model = 'mistralai/Mistral-7B-Instruct-v0.2'
model = 'meta-llama/Meta-Llama-3-8B-Instruct'
# model = 'google/gemma-1.1-7b-it'
# model = 'google/gemma-1.1-2b-it'
# model = 'microsoft/Phi-3-mini-4k-instruct'

# model = 'meta-llama/Llama-2-7b-chat-hf'
lora = ''

# part = 'AI4Chem'
part = 'low'
if part == 'high':
    if lora != '':
        for i in range(60):
            os.system(f'sbatch -s --overcommit --requeue -p vip_gpu_ailab -A ai4phys --gres=gpu:1 ./server.bash {model} {10000+i} {lora}')
    else:
        for i in range(60):
            os.system(f'sbatch -s --overcommit --requeue -p vip_gpu_ailab -A ai4phys --gres=gpu:1 ./server.bash {model} {10000+i}')

elif part == 'low':
    if lora != '':
        for i in range(16):
            os.system(f'sbatch -s --overcommit --requeue -p vip_gpu_ailab_low --gres=gpu:1 ./server.bash {model} {10000+i} {lora}')
    else:
        for i in range(16):
            os.system(f'sbatch -s --overcommit --requeue -p vip_gpu_ailab_low --gres=gpu:1 ./server.bash {model} {10000+i}')
