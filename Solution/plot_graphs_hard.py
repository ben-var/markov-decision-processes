import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

outdir = 'out/plots/hard/'

lr_test = False
Q_test = False

vi_hard_f = 'out/hard/refined run/Hard Value.csv'
pi_hard_f = 'out/hard/refined run/Hard Policy.csv'
best_Q = 'out/hard/refined run/Hard Q-Learning L0.5 q100.0 E0.5.csv'

evi = pd.read_csv(vi_hard_f)
epi = pd.read_csv(pi_hard_f)
eQ = pd.read_csv(best_Q)

algos = {'VI': evi, 'PI': epi,'Q': eQ}

plt.title('HardGW Convergence over Iterations')

for key, val in algos.items():
    df = algos[key]
    plt.plot(df['iter'], df['convergence'], label=key)
plt.legend()
plt.xlim([0,200])
plt.ylabel('Convergence Delta')
plt.xlabel('Iterations')
plt.savefig(outdir + 'convergence_summary.png')
plt.close()

plt.title('HardGW Time over Iterations')

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

plt.title('HardGW Time over Iterations Q-Learning')
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

plt.title('HardGW Total Reward over Iterations')
for key, val in algos.items():
    if key is 'Q':
        continue
    df = algos[key]
    plt.plot(df['iter'], df['reward'], label=key)
plt.legend()
plt.ylabel('Total Reward')
plt.xlabel('Iterations')
plt.xlim([0,120])
plt.savefig(outdir + 'reward_qi_pi.png')
plt.close()

plt.title('HardGW Total Reward over Iterations Q-Learning')
for key, val in algos.items():
    if key is not 'Q':
        continue
    df = algos[key]
    plt.plot(df['iter'], df['reward'], label=key)
plt.legend()
plt.ylabel('Total Reward')
plt.xlabel('Iterations')
plt.xlim([-300,10000])
plt.savefig(outdir + 'reward_q.png')
plt.close()


if lr_test:

    lr = 'L0.1'

    eQ_files = []
    for subdir, dirs, files in os.walk('out/hard/initial/Q/'):
        for file in files:
            filepath = subdir + os.sep + file

            if filepath.endswith(".csv") and lr in filepath:
                eQ_files.append(filepath)

    for f in eQ_files:

        eQ = pd.read_csv(f)
        lab = f.upper()[41:-4]
        plt.plot(eQ['iter'], eQ['convergence'], label=lab)

    plt.title('HardGW Convergence over Iterations - Learning Rate of ' + lr)
    plt.ylabel('Convergence Delta')
    plt.xlabel('Iterations')
    plt.legend(loc='best')
    plt.savefig(outdir + 'lr0.1.png')
    plt.close()

    lr = 'L0.5'

    eQ_files = []
    for subdir, dirs, files in os.walk('out/hard/initial/Q/'):
        for file in files:
            filepath = subdir + os.sep + file

            if filepath.endswith(".csv") and lr in filepath:
                eQ_files.append(filepath)

    for f in eQ_files:

        eQ = pd.read_csv(f)
        lab = f.upper()[41:-4]
        plt.plot(eQ['iter'], eQ['convergence'], label=lab)

    plt.title('HardGW Convergence over Iterations - Learning Rate of ' + lr)
    plt.ylabel('Convergence Delta')
    plt.xlabel('Iterations')
    plt.legend(loc='best')
    plt.savefig(outdir + 'lr0.5.png')
    plt.close()


    lr = 'L0.9'

    eQ_files = []
    for subdir, dirs, files in os.walk('out/hard/initial/Q/'):
        for file in files:
            filepath = subdir + os.sep + file

            if filepath.endswith(".csv") and lr in filepath:
                eQ_files.append(filepath)

    for f in eQ_files:

        eQ = pd.read_csv(f)
        lab = f.upper()[41:-4]
        plt.plot(eQ['iter'], eQ['convergence'], label=lab)

    plt.title('HardGW Convergence over Iterations - Learning Rate of ' + lr)
    plt.legend(loc='best')
    plt.ylabel('Convergence Delta')
    plt.xlabel('Iterations')
    plt.savefig(outdir + 'lr0.9.png')
    plt.close()


if Q_test:
    lr = 'L0.5'
    initQ = 'q0.0'

    eQ_files = []
    for subdir, dirs, files in os.walk('out/hard/initial/Q/'):
        for file in files:
            filepath = subdir + os.sep + file

            if filepath.endswith(".csv") and lr in filepath and initQ in filepath:
                eQ_files.append(filepath)

    for f in eQ_files:

        eQ = pd.read_csv(f)
        lab = f.upper()[41:-4]
        plt.plot(eQ['iter'], eQ['convergence'], label=lab)

    plt.title('HardGW Convergence over Iterations - inital Q of ' + initQ)
    plt.legend(loc='best')
    plt.ylabel('Convergence Delta')
    plt.xlabel('Iterations')
    plt.savefig(outdir + 'q0.png')
    plt.close()

    initQ = 'q-100'

    eQ_files = []
    for subdir, dirs, files in os.walk('out/hard/initial/Q/'):
        for file in files:
            filepath = subdir + os.sep + file

            if filepath.endswith(".csv") and lr in filepath and initQ in filepath:
                eQ_files.append(filepath)

    for f in eQ_files:

        eQ = pd.read_csv(f)
        lab = f.upper()[41:-4]
        plt.plot(eQ['iter'], eQ['convergence'], label=lab)

    plt.title('HardGW Convergence over Iterations - inital Q of ' + initQ)
    plt.legend(loc='best')
    plt.ylabel('Convergence Delta')
    plt.xlabel('Iterations')
    plt.savefig(outdir + 'q-100.png')
    plt.close()

    initQ = 'q100'

    eQ_files = []
    for subdir, dirs, files in os.walk('out/hard/initial/Q/'):
        for file in files:
            filepath = subdir + os.sep + file

            if filepath.endswith(".csv") and lr in filepath and initQ in filepath:
                eQ_files.append(filepath)

    for f in eQ_files:

        eQ = pd.read_csv(f)
        lab = f.upper()[41:-4]
        plt.plot(eQ['iter'], eQ['convergence'], label=lab)

    plt.title('HardGW Convergence over Iterations - inital Q of ' + initQ)
    plt.legend(loc='best')
    plt.ylabel('Convergence Delta')
    plt.xlabel('Iterations')
    plt.savefig(outdir + 'q100.png')
    plt.close()

lr = 'L0.5'

eQ_files = []
for subdir, dirs, files in os.walk('out/hard/initial/Q/'):
    for file in files:
        filepath = subdir + os.sep + file

        if filepath.endswith(".csv") and lr in filepath:
            eQ_files.append(filepath)

for f in eQ_files:

    eQ = pd.read_csv(f)
    lab = f.upper()[41:-4]
    plt.plot(eQ['iter'], eQ['convergence'], label=lab)

plt.title('HardGW Convergence over Iterations - Learning Rate of ' + lr)
plt.legend(loc='best')
plt.ylabel('Convergence Delta')
plt.xlabel('Iterations')
plt.savefig(outdir + 'best_lr_summary.png')
plt.close()
