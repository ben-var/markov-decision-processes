import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

outdir = 'out/plots/easy/'

lr_test = False
Q_test = False

vi_easy_f = 'out/easy/initial/Easy Value.csv'
pi_easy_f = 'out/easy/initial/Easy Policy.csv'
best_Q = 'out/easy/Easy Q-Learning L0.1 q0.0 E0.1.csv'

evi = pd.read_csv(vi_easy_f)
epi = pd.read_csv(pi_easy_f)
eQ = pd.read_csv(best_Q)

algos = {'VI': evi, 'PI': epi,'Q': eQ}

plt.title('EasyGW Convergence over Iterations')

for key, val in algos.items():
    df = algos[key]
    plt.plot(df['iter'], df['convergence'], label=key)
plt.legend()
plt.xlim([0,50])
plt.ylabel('Convergence Delta')
plt.xlabel('Iterations')
plt.savefig(outdir + 'convergence_summary.png')
plt.close()

plt.title('EasyGW Convergence over Iterations (Zoomed)')

for key, val in algos.items():
    df = algos[key]
    plt.plot(df['iter'], df['convergence'], label=key)
plt.legend()
plt.ylim([0,10])
plt.xlim([0,50])
plt.ylabel('Convergence Delta')
plt.xlabel('Iterations')
plt.savefig(outdir + 'convergence_summary_zoom.png')
plt.close()

plt.title('EasyGW Time over Iterations')

for key, val in algos.items():
    if key is 'Q':
        continue
    df = algos[key]
    plt.plot(df['iter'], df['time'], label=key)
plt.legend()
plt.ylabel('Wall Clock Time (s)')
plt.xlabel('Iterations')
plt.savefig(outdir + 'time_pi_vi.png')
plt.close()

plt.title('EasyGW Time over Iterations Q-Learning')
for key, val in algos.items():
    if key is not 'Q':
        continue
    df = algos[key]
    plt.plot(df['iter'], df['time'], label=key)
plt.legend()
plt.ylabel('Wall Clock Time (s)')
plt.xlabel('Iterations')
plt.savefig(outdir + 'time_q.png')
plt.close()

plt.title('EasyGW Total Reward over Iterations')
for key, val in algos.items():
    if key is 'Q':
        continue
    df = algos[key]
    plt.plot(df['iter'], df['reward'], label=key)
plt.legend()
plt.ylabel('Total Reward')
plt.xlabel('Iterations')
plt.savefig(outdir + 'reward_qi_pi.png')
plt.close()

plt.title('EasyGW Total Reward over Iterations Q-Learning')
for key, val in algos.items():
    if key is not 'Q':
        continue
    df = algos[key]
    plt.plot(df['iter'], df['reward'], label=key)
plt.legend()
plt.ylabel('Total Reward')
plt.xlabel('Iterations')
plt.xlim([-300, 3000])
plt.savefig(outdir + 'reward_q.png')
plt.close()

plt.title('EasyGW Total Reward over Iterations Q-Learning (Zoomed)')
for key, val in algos.items():
    if key is not 'Q':
        continue
    df = algos[key]
    plt.plot(df['iter'], df['reward'], label=key)
plt.legend()
plt.ylabel('Total Reward')
plt.xlabel('Iterations')
plt.xlim([-5, 100])
plt.savefig(outdir + 'reward_q_zoom.png')
plt.close()


if lr_test:

    lr = 'L0.1'

    eQ_files = []
    for subdir, dirs, files in os.walk('out/easy/initial/Q/'):
        for file in files:
            filepath = subdir + os.sep + file

            if filepath.endswith(".csv") and lr in filepath:
                eQ_files.append(filepath)

    for f in eQ_files:

        eQ = pd.read_csv(f)
        lab = f.upper()[41:-4]
        plt.plot(eQ['iter'], eQ['convergence'], label=lab)

    plt.title('EasyGW Convergence over Iterations - Learning Rate of ' + lr)
    plt.ylim([0, 20])
    plt.ylabel('Convergence Delta')
    plt.xlabel('Iterations')
    plt.legend(loc='best')
    plt.savefig(outdir + 'lr0.1.png')
    plt.close()


    lr = 'L0.3'

    eQ_files = []
    for subdir, dirs, files in os.walk('out/easy/initial/Q/'):
        for file in files:
            filepath = subdir + os.sep + file

            if filepath.endswith(".csv") and lr in filepath:
                eQ_files.append(filepath)

    for f in eQ_files:

        eQ = pd.read_csv(f)
        lab = f.upper()[41:-4]
        plt.plot(eQ['iter'], eQ['convergence'], label=lab)

    plt.title('EasyGW Convergence over Iterations - Learning Rate of ' + lr)
    plt.ylim([0,40])
    plt.ylabel('Convergence Delta')
    plt.xlabel('Iterations')
    plt.legend(loc='best')
    plt.savefig(outdir + 'lr0.3.png')
    plt.close()

    lr = 'L0.5'

    eQ_files = []
    for subdir, dirs, files in os.walk('out/easy/initial/Q/'):
        for file in files:
            filepath = subdir + os.sep + file

            if filepath.endswith(".csv") and lr in filepath:
                eQ_files.append(filepath)

    for f in eQ_files:

        eQ = pd.read_csv(f)
        lab = f.upper()[41:-4]
        plt.plot(eQ['iter'], eQ['convergence'], label=lab)

    plt.title('EasyGW Convergence over Iterations - Learning Rate of ' + lr)
    plt.ylim([0,60])
    plt.ylabel('Convergence Delta')
    plt.xlabel('Iterations')
    plt.legend(loc='best')
    plt.savefig(outdir + 'lr0.5.png')
    plt.close()


    lr = 'L0.9'

    eQ_files = []
    for subdir, dirs, files in os.walk('out/easy/initial/Q/'):
        for file in files:
            filepath = subdir + os.sep + file

            if filepath.endswith(".csv") and lr in filepath:
                eQ_files.append(filepath)

    for f in eQ_files:

        eQ = pd.read_csv(f)
        lab = f.upper()[41:-4]
        plt.plot(eQ['iter'], eQ['convergence'], label=lab)

    plt.title('EasyGW Convergence over Iterations - Learning Rate of ' + lr)
    plt.legend(loc='best')
    plt.ylabel('Convergence Delta')
    plt.xlabel('Iterations')
    plt.savefig(outdir + 'lr0.9.png')
    plt.close()


if Q_test:
    lr = 'L0.1'
    initQ = 'q0.0'

    eQ_files = []
    for subdir, dirs, files in os.walk('out/easy/initial/Q/'):
        for file in files:
            filepath = subdir + os.sep + file

            if filepath.endswith(".csv") and lr in filepath and initQ in filepath:
                eQ_files.append(filepath)

    for f in eQ_files:

        eQ = pd.read_csv(f)
        lab = f.upper()[41:-4]
        plt.plot(eQ['iter'], eQ['convergence'], label=lab)

    plt.title('EasyGW Convergence over Iterations - inital Q of ' + initQ)
    plt.ylim([0, 12])
    plt.legend(loc='best')
    plt.ylabel('Convergence Delta')
    plt.xlabel('Iterations')
    plt.savefig(outdir + 'q0.png')
    plt.close()

    lr = 'L0.1'
    initQ = 'q-100'

    eQ_files = []
    for subdir, dirs, files in os.walk('out/easy/initial/Q/'):
        for file in files:
            filepath = subdir + os.sep + file

            if filepath.endswith(".csv") and lr in filepath and initQ in filepath:
                eQ_files.append(filepath)

    for f in eQ_files:

        eQ = pd.read_csv(f)
        lab = f.upper()[41:-4]
        plt.plot(eQ['iter'], eQ['convergence'], label=lab)

    plt.title('EasyGW Convergence over Iterations - inital Q of ' + initQ)
    plt.ylim([0, 12])
    plt.legend(loc='best')
    plt.ylabel('Convergence Delta')
    plt.xlabel('Iterations')
    plt.savefig(outdir + 'q-100.png')
    plt.close()

    lr = 'L0.1'
    initQ = 'q100'

    eQ_files = []
    for subdir, dirs, files in os.walk('out/easy/initial/Q/'):
        for file in files:
            filepath = subdir + os.sep + file

            if filepath.endswith(".csv") and lr in filepath and initQ in filepath:
                eQ_files.append(filepath)

    for f in eQ_files:

        eQ = pd.read_csv(f)
        lab = f.upper()[41:-4]
        plt.plot(eQ['iter'], eQ['convergence'], label=lab)

    plt.title('EasyGW Convergence over Iterations - inital Q of ' + initQ)
    plt.ylim([0, 12])
    plt.legend(loc='best')
    plt.ylabel('Convergence Delta')
    plt.xlabel('Iterations')
    plt.savefig(outdir + 'q100.png')
    plt.close()

lr = 'L0.1'

eQ_files = []
for subdir, dirs, files in os.walk('out/easy/initial/Q/'):
    for file in files:
        filepath = subdir + os.sep + file

        if filepath.endswith(".csv") and lr in filepath:
            eQ_files.append(filepath)

for f in eQ_files:

    eQ = pd.read_csv(f)
    lab = f.upper()[41:-4]
    if (eQ['convergence'].min() < 1.6):
        plt.plot(eQ['iter'], eQ['convergence'], label=lab)

plt.title('EasyGW Convergence over Iterations - Learning Rate of ' + lr)
plt.ylim([0, 20])
plt.legend(loc='best')
plt.ylabel('Convergence Delta')
plt.xlabel('Iterations')
plt.savefig(outdir + 'best_lr_summary.png')
plt.close()
