import os

bn1 = 'CNN1'
bn2 = 'CNN2'

srcdir = './data/test/input'
dstdir = './data/test/output'

if not os.path.exists(dstdir):
    os.mkdir(dstdir)

l1 = f'{srcdir}/{bn1}.canonical_lmks'
l2 = f'{srcdir}/{bn2}.canonical_lmks'

v1 = f'{srcdir}/{bn1}.mp4'
v2 = f'{srcdir}/{bn2}.mp4'

r1 = f'{srcdir}/{bn1}.face_rects'
r2 = f'{srcdir}/{bn2}.face_rects'

model_bn = 'cnn1d-v2-w30'

outpath = f'{dstdir}/CNN_{model_bn}.mp4'

cmd = f'python VAD_dyadic.py {l1} {l2} {v1} {v2} {outpath} --rects1_path={r1} --rects2_path={r2} --model_fpath=./models/{model_bn}.pth'

print(cmd)
os.system(cmd)


