import os, sys, math
import multiprocessing

# use vips to slice WSI to pathes, keep 20x patchs only
def dzsave(files, in_dir, out_dir, patch_size):
    for f in files:
        os.system('vips dzsave %s %s --tile-size %d' % (os.path.join(in_dir, f), os.path.join(out_dir, f[:-4]), patch_size))
        os.system('sleep 3s')

        root = os.path.join(out_dir, f[:-4] + '_files')
        ds = []
        for _ in os.listdir(root):
            if '.xml' not in _:
                ds.append(int(_))
        ds.sort()

        # keep x20 only for camelyon and tcga [-1] is x40
        keep = ds[-2]
        for d in ds:
            if d != keep:
                os.system('rm -rf ' + os.path.join(root, str(d)))

def main():

    in_dir = sys.argv[1]
    out_dir = sys.argv[2]
    worker_num = int(sys.argv[3])
    patch_size = int(sys.argv[4])

    os.makedirs(out_dir, exist_ok=True)
    files = os.listdir(in_dir)
    temp = []
    for f in files:
        if ('.svs' in f  or '.tif' in f) and f.replace('.svs', '.dzi').replace('.tif', '.dzi') not in os.listdir(out_dir):
            temp.append(f)
    files = temp

    num_each = math.ceil(len(files) / worker_num)
    worker_num = math.ceil(len(files) / num_each)

    pool = multiprocessing.Pool(processes = worker_num)
    for i in range(worker_num):
        start = i * num_each
        end = i * num_each + num_each
        if start < len(files):
            pool.apply_async(dzsave, (files[start: end], in_dir, out_dir, patch_size))
    pool.close()
    pool.join()


if __name__ == '__main__':
    main()
