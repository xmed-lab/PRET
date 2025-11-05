import os, sys, time, random
import subprocess
import multiprocessing
from run import get_run_command


# ====================== gpu management ======================

# get free gpu memory
def get_gpu_memory(gpu_index):
    result = subprocess.run(['nvidia-smi', '--id={}'.format(gpu_index), '--query-gpu=memory.free', '--format=csv,nounits,noheader'], capture_output=True, text=True)
    gpu_memory_info = result.stdout.strip().split('\n')
    gpu_memory_free = int(gpu_memory_info[0])
    return gpu_memory_free

# get tasks on specific gpu
def get_running_processes_count(gpu_index):
    command = 'nvidia-smi --id={} --query-compute-apps=pid --format=csv,noheader'.format(gpu_index)
    output = subprocess.check_output(command, shell=True, encoding='utf-8')
    processes = output.strip().split('\n')
    return len(processes)

# get available gpus to run multiple tasks
def get_available_gpus(devices, memory_limitation, max_tasks=2):
    gpus = []
    for gpu_index in devices:
        rest_memory = get_gpu_memory(gpu_index)
        running_tasks = get_running_processes_count(gpu_index)
        if rest_memory > memory_limitation and running_tasks < max_tasks:
            gpus.append(gpu_index)
    return gpus

# run the command for a single task
def execute_bash_command(command, output_file):
    try:
        with open(output_file, 'w') as file:
            subprocess.run(command, shell=True, check=True, stdout=file, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        print(f"wrong command: {command}")
        print(f"error info: {e.output}")


if __name__ == '__main__':

    # ====================== set inputs =====================

    # which used dataset, type the path
    dataset = sys.argv[1]

    # which model to use, type the path or name
    model = sys.argv[2]

    # how many tasks run at one moment
    num_processer = sys.argv[3]
    
    # gpu settings, ids, memory (MB), max task number per gpu
    visible_devices = [1,2,3,4]
    gpu_memory_limitation = 8000
    max_tasks_per_gpu = 3

    # ===================== extract features ======================

    # prepare folder and parse model name for feature_extractor
    FEAT_DIR = 'data/' + dataset + '/patch/features'
    PATCH_DIR = 'data/' + dataset + '/patch/images'
    if 'model' in model:
        arch = 'vit_small'
    else:
        FEAT_DIR += '_' + model.split('/')[-1].split('.')[0]
        arch = model.split('/')[-1].split('.')[0]

    # extract feature by invoking the core/feature_extractor.py
    if not os.path.exists(FEAT_DIR):
        command = 'CUDA_VISIBLE_DEVICES=' + ','.join(map(str, visible_devices)) + ' python -u -m torch.distributed.launch --nproc_per_node=' \
            + str(len(visible_devices)) + ' --master_port=2333 core/feature_extractor.py --data_path ' + PATCH_DIR + ' --pretrained_weights ' \
            + model + ' --dump_features ' + FEAT_DIR + ' --num_workers 8 --batch_size_per_gpu 32 --arch ' + arch
        os.system(command)

    # extract feature only
    if num_processer == 0:
        os._exit(0)

    # ===================== generate multiple commands ======================

    # generate commands multiple tasks and prompt types for a dataset
    commands = []
    if dataset in ['GC', 'LC', 'PTC', 'ESCC', 'CRC', 'BC', 'PTC_QP']:
        tasks = ['screening', 'segmentation']
    elif dataset in ['CAMELYON16', 'CAMELYON16C', 'CAMELYON17']:
        tasks = ['lnm']
    elif dataset in ['NSCLC', 'ESCA', 'SARC', 'RCC', 'LN', 'NSCLC_HQ']:
        tasks = ['subtyping']
    for task in tasks:
        for mode in ['baselines', 'default']:
            for prompt_type in ['slideLabel', 'mask', 'box', 'roughMask']:
                if 'CAMELYON' in dataset and prompt_type != 'mask':
                    continue
                # invoke the scripts/run.py for the command for each benchmarks
                commands.append(get_run_command(dataset, task, mode, prompt_type, model))


    # ===================== run commands for tasks and prompts ======================

    # run multiple commands in a multiprocessing Pool
    pool = multiprocessing.Pool(processes=int(num_processer))
    for command in commands:
        if command == '':
            continue
        not_run = True
        while not_run:
            available_gpus = get_available_gpus(visible_devices, gpu_memory_limitation, max_tasks_per_gpu)
            if available_gpus:
                gpu_index = available_gpus[random.randint(0, len(available_gpus) - 1)]
                command = 'CUDA_VISIBLE_DEVICES=' + str(gpu_index) + ' ' + command
                command, out_text = command.split(' &> ')
                pool.apply_async(execute_bash_command, args=(command, out_text), kwds={}, callback=None)
                not_run = False

            # waiting GPU allocation to avoid too much tasks in one GPU
            time.sleep(300)

    pool.close()
    pool.join()
