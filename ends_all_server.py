import os
table = os.popen('squeue').read()
for line in str(table).split('\n'):
    # print(line)
    if 'server.b' in line or 'batch.sh' in line:
        id = line.split('vip_gpu')[0] #change to your slurm partition name
        os.system(f'scancel {id}')
# print(table)
try:
    os.remove('./server.csv')
except:
    pass
try:
    os.system('rm *.out')
except:
    pass
os.system('squeue')