import sys, json, os, cv2

wsi_dir = sys.argv[1]
gt_dir = sys.argv[2]
label_fn = sys.argv[3]
outf = sys.argv[4]


label = {}
for l in open(label_fn).readlines():
    tks = l.strip().split(',')
    label[tks[0]] = int(tks[1])

out_json = {}

for f in label.keys():
    n = f[:-4]
    out_json[n] = {}
    if n + '.png' in os.listdir(gt_dir):
        out_json[n]['patch_labels'] = os.path.join(gt_dir, n + '.png')
        gt = cv2.imread(out_json[n]['patch_labels'])[:, :, 0]
        out_json[n]['pos_patch_num'] = int((gt == 1).sum())
    out_json[n]['wsi_label'] = label[f]
    out_json[n]['fixed_test_set'] = False

outs = json.dumps(out_json, indent=4)
open(outf, 'w').write(outs)
