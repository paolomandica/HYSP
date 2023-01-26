import pickle
import numpy as np
import itertools
from tqdm import tqdm
from joblib import Parallel, delayed

N_JOBS = 32
RANDOM_SEARCH = True
RANDOM_FACTOR = 8
ALPHA = None  # [0.9, 0.5, 0.6]  # joint, bone, motion

# Linear
print('-' * 20 + '3-stream Eval' + '-' * 20)

joint_path = 'checkpoints/hysp_xview_joint/supervised/score_list.npy'
bone_path = 'checkpoints/hysp_xview_bone/supervised/score_list.npy'
motion_path = 'checkpoints/hysp_xview_motion/supervised/score_list.npy'

label = open('data/ntu60_frame50/xview/val_label.pkl', 'rb')
label = np.array(pickle.load(label))

r1 = np.load(joint_path)
r2 = np.load(bone_path)
r3 = np.load(motion_path)


def compute_accuracy(alpha):
    right_num = total_num = right_num_5 = 0
    for i in range(r1.shape[0]):
        _, l = label[:, i]
        r11 = r1[i]
        r22 = r2[i]
        r33 = r3[i]
        r = r11 * alpha[0] + r22 * alpha[1] + r33 * alpha[2]
        rank_5 = r.argsort()[-5:]
        right_num_5 += int(int(l) in rank_5)
        r = np.argmax(r)
        right_num += int(r == int(l))
        total_num += 1
    acc = right_num / total_num
    acc5 = right_num_5 / total_num
    return (acc, acc5), alpha


if ALPHA is None:
    alpha_seq = [i/10 for i in range(0, 10)]
    alpha_combinations = list(itertools.combinations(
        itertools.chain(alpha_seq, alpha_seq, alpha_seq), 3))

    if RANDOM_SEARCH:
        idx = np.random.choice(len(alpha_combinations), len(
            alpha_combinations)//RANDOM_FACTOR, replace=False)
        alpha_combinations = [alpha_combinations[i] for i in idx]

    results = Parallel(n_jobs=N_JOBS)(delayed(compute_accuracy)(alpha)
                                      for alpha in tqdm(alpha_combinations))
    (acc, acc5), best_alpha = max(results, key=lambda x: x[0][0])

else:
    (acc, acc5), best_alpha = compute_accuracy(ALPHA)

print('\ntop1: ', round(acc, 4)*100)
print('top5: ', round(acc5, 4)*100)
print('best_alpha: ', best_alpha)
print()
