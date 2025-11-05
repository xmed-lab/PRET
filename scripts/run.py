import os, sys


# ===================== generate a single command ======================

def get_run_command(dataset, task, mode, prompt_type, model):

    # ===================== set folders ======================

    # basic patches
    WSI_DIR = 'data/' + dataset + '/images'
    FEAT_DIR = 'data/' + dataset + '/patch/features'
    COLLECTED_FEAT_DIR = 'data/' + dataset + '/patch/collected_features'
    PROMPT_DIR = 'data/' + dataset + '/anno'
    RECORDS = 'records/%s_%s_%s_%s' % (dataset, task, prompt_type, mode)

    # path for local model weights or name for huggingface models
    if 'model' not in model:
        model_name = model.split('/')[-1].split('.')[0]
        FEAT_DIR += '_' + model_name
        COLLECTED_FEAT_DIR += '_' + model_name
        RECORDS += '_' + model_name

    # ===================== set hyper-parameters ======================

    # basic hyper-params
    SEED = 1024
    TOP_INS = 3000 if 'CAMELYON' not in dataset else 1
    FILE_MIN_SIZE = 5000 if prompt_type != 'slideLabel' else 15000
    TOPK = 40
    TEMP = 10
    THRESH = 0.88
    IGNORE = 0
    IGNORE_QUERY = 0.2
    SHOT_NUM = 8
    CLS_NUM = 1 if task in ['screening', 'segmentation', 'lnm'] else 2
    
    # ===================== set data split ======================

    # data split principle
    # 1) TCGA 20% test similating 5-fold, 100 val or 50 val if not sufficient
    # 2) New datasets is smaller than TCGA, thus 1/3 val 2/3 test for slides out of example
    # 3) Cross race NSCLC, White follow TCGA split, others follow new datasets
    TEST_NUM, VAL_RATIO = -1, -1
    VAL_NUM = 100
    if dataset in ['BC', 'CRC', 'ESCC', 'GC', 'LC', 'PTC', 'PTC_QP', 'NSCLC_HQ'] or task == 'segmentation':
        VAL_RATIO = 0.3334
    elif dataset == 'NSCLC':
        TEST_NUM = 210
    elif dataset == 'RCC':
        TEST_NUM = 188
        CLS_NUM = 3
    elif dataset == 'ESCA':
        TEST_NUM = 30
        VAL_NUM = 50
    elif dataset == 'SARC':
        TEST_NUM = 69
        CLS_NUM = 3
    elif dataset == 'LN':
        FILE_MIN_SIZE=5000
        VAL_RATIO = 0.3334
    if dataset == 'PTC_QP':
        FILE_MIN_SIZE=5000
    if FILE_MIN_SIZE == 15000:
        COLLECTED_FEAT_DIR += '_size15k'

    # ===================== generate the command ======================

    # prepare args for the core/main.py
    command = 'python -u core/main.py --mode ' + mode + \
            ' --topk ' + str(TOPK) + ' --temperature ' + str(TEMP) + ' --related_thresh ' + str(THRESH) + \
            ' --example_num ' + str(SHOT_NUM) + ' --raw_feature_path ' + FEAT_DIR + ' --wsi_path '  + WSI_DIR + \
            ' --dump_features ' + COLLECTED_FEAT_DIR + ' --dataset_info data_info/' + dataset + '.json ' + \
            ' --seed ' + str(SEED) + ' --top_instance ' + str(TOP_INS) + ' --test_num ' + str(TEST_NUM) + \
            ' --val_num ' + str(VAL_NUM) + ' --val_ratio ' + str(VAL_RATIO) + ' --prompt_type ' + prompt_type + \
            ' --prompt_path ' + PROMPT_DIR + ' --ignore ' + str(IGNORE) + ' --multiple_num 1 2 4 8 ' + \
            ' --file_min_size ' + str(FILE_MIN_SIZE) + ' --c ' + str(CLS_NUM) + ' --ignore_query ' + \
            str(IGNORE_QUERY) + ' --dump_records ' + RECORDS + '.npy'
    if task == 'segmentation':
        command += ' --seg '

    # save the records in both .txt and .npy formats
    command += ' &> ' + RECORDS + '.txt'

    # skip it, if already done
    if os.path.exists(RECORDS + '.npy'):
        command = ''

    return command


# ===================== run a singel task and prompt if needed ======================

if __name__ == '__main__':
    gpu_id = sys.argv[1]
    dataset = sys.argv[2]
    task = sys.argv[3]
    mode = sys.argv[4]
    prompt_type = sys.argv[5]
    model = sys.argv[6]

    command = get_run_command(dataset, task, mode, prompt_type, model)
    command = 'CUDA_VISIBLE_DEVICES=' + gpu_id + ' ' + command
    print(command)
    os.system(command)
