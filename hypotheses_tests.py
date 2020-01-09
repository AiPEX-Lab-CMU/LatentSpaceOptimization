import os,sys
import numpy as np
from scipy.stats import ttest_ind_from_stats as ttest
from scipy.spatial.distance import directed_hausdorff
import generate
import pickle

NUM_PARAMS = 1024
SEED = 10

def get_pop_similarity(pop):
    generator = generate.Generator(num_params=NUM_PARAMS)
    pop_points = []
    for vec in pop:
        points = generator.generate_return_pts(vec)
        pop_points.append(points)
    hsums = []
    for i in range(len(pop_points)):
        for j in range(i,len(pop_points)):
            hsums.append(directed_hausdorff(pop_points[i],pop_points[j])[0])
    return np.mean(hsums)

def get_trainset_similarity(pop):
    generator = generate.Generator(num_params=NUM_PARAMS)
    #Get GA population point set
    pop_points = []
    for vec in pop:
        points = generator.generate_return_pts(vec)
        pop_points.append(points)

    #Get Training Set point set
    train_vecs, _ = generator.load_training_objects(300,SEED)
    train_points = []
    for vec in train_vecs:
        points = generator.generate_return_pts(vec)
        train_points.append(points)

    hsums = []
    for ppoints in pop_points:
        for tpoints in train_points:
            hsums.append(directed_hausdorff(ppoints,tpoints)[0])
    return np.mean(hsums)


def H1_testing(exp_names,trials,load_dir):
    write_file = 'hLogs/H1.txt'
    for name in exp_names:
        init_scores = []
        final_scores = []
        pct_improves = []
        for i in range(trials):
            load_file = load_dir + name + str(i) + '.pkl'
            with open(load_file,'rb') as f:
                if 'ST' in name:
                    _,_,learn_curve,_,_,_,_,_ = pickle.load(f)
                else:
                    _,learn_curve,_,_,_,_,_ = pickle.load(f)
            init_score = learn_curve[0]
            final_score = learn_curve[-1]
            diff_score = final_score-init_score
            pct_improve = diff_score/init_score

            init_scores.append(init_score)
            final_scores.append(final_score)
            pct_improves.append(pct_improve)

        mean_init = np.mean(init_scores)
        std_init = np.std(init_scores)
        mean_final = np.mean(final_scores)
        std_final = np.std(final_scores)
        mean_pct_improve = -np.mean(pct_improve)
        std_pct_improve = np.std(pct_improve)
        p_diff = ttest(mean_final, std_final, trials, mean_init, std_init, trials)

        with open(write_file,'a+') as f:
            f.write(name + '\n')
            f.write('Mean Initial Score: {} \n'.format(mean_init))
            f.write('STD Initial Score: {} \n'.format(std_init))
            f.write('Mean Final Score: {} \n'.format(mean_final))
            f.write('STD Final Score: {} \n'.format(std_final))
            f.write('Mean Percent Improvement: {} \n'.format(mean_pct_improve))
            f.write('STD Percent Improvement: {} \n'.format(std_pct_improve))
            f.write('Pval Inital-Final Difference: {} \n'.format(p_diff))
            f.write('\n')

def H23_testing(exp_names,trials,load_dir):
    h2_write_file = 'hLogs/H2.txt'
    h3_write_file = 'hLogs/H3.txt'

    mean_sims = []
    std_sims = []
    for name in exp_names:
        sims = []
        for i in range(trials):
            print(i)
            load_file = load_dir + name + str(i) + '.pkl'
            with open(load_file,'rb') as f:
                    if 'ST' in name:
                        pop,_,_,_,_,_,_,_ = pickle.load(f)
                    else:
                        pop,_,_,_,_,_,_ = pickle.load(f)
            sim = get_pop_similarity(pop)
            sims.append(sim)
        mean_sim = np.mean(sims)
        mean_sims.append(mean_sim)
        std_sim = np.std(sims)
        std_sims.append(std_sim)

        if 'DIF_SF' in name:
            mean_sim_naive = mean_sim
            std_sim_naive = std_sim
        if 'DIT_SF' in name:  
            mean_sim_tdi = mean_sim
            std_sim_tdi = std_sim

    for i in range(len(exp_names)):
        name = exp_names[i]
        mean_sim = mean_sims[i]
        std_sim = std_sims[i]
        if 'DIF' in name:
            with open(h2_write_file,'a+') as f:
                f.write(name + '\n')
                f.write('Mean Hausdorffs: {} \n'.format(mean_sim))
                f.write('STD Hausdorffs: {} \n'.format(std_sim))
                f.write('\n')
        if 'DIT_SF' in name: #compare to naive (H2) and write to (H3)
            pct_improve_naive = (mean_sim - mean_sim_naive)/mean_sim_naive
            pval_naive = ttest(mean_sim, std_sim, trials, mean_sim_naive, std_sim_naive, trials)
            with open(h2_write_file,'a+') as f:
                f.write(name + '\n')
                f.write('Mean Hausdorffs: {} \n'.format(mean_sim))
                f.write('STD Hausdorffs: {} \n'.format(std_sim))
                f.write('Mean Percent Improvement: {} \n'.format(pct_improve_naive))
                f.write('Pval Naive: {} \n'.format(pval_naive))
                f.write('\n')
            with open(h3_write_file,'a+') as f:
                f.write(name + '\n')
                f.write('Mean Hausdorffs: {} \n'.format(mean_sim))
                f.write('STD Hausdorffs: {} \n'.format(std_sim))
                f.write('\n')
        if 'ST' in name: #compare to DIF_SF and DIT_SF (H3)
            pct_improve_naive = (mean_sim - mean_sim_naive)/mean_sim_naive
            pct_improve_tdi = (mean_sim - mean_sim_tdi)/mean_sim_tdi
            pval_naive = ttest(mean_sim, std_sim, trials, mean_sim_naive, std_sim_naive, trials)
            pval_tdi = ttest(mean_sim, std_sim, trials, mean_sim_tdi, std_sim_tdi, trials)
            with open(h2_write_file,'a+') as f:
                f.write(name + '\n')
                f.write('Mean Hausdorffs: {} \n'.format(mean_sim))
                f.write('STD Hausdorffs: {} \n'.format(std_sim))
                f.write('Mean Percent Improvement: {} \n'.format(pct_improve_naive))
                f.write('Pval TDI: {} \n'.format(pval_naive))
                f.write('\n')
            with open(h3_write_file,'a+') as f:
                f.write(name + '\n')
                f.write('Mean Hausdorffs: {} \n'.format(mean_sim))
                f.write('STD Hausdorffs: {} \n'.format(std_sim))
                f.write('Mean Percent Improvement: {} \n'.format(pct_improve_tdi))
                f.write('Pval TDI: {} \n'.format(pval_tdi))
                f.write('\n')

def H4_testing(exp_names,trials,load_dir):
    write_file = 'hLogs/H4.txt'
    mean_sims = []
    std_sims = []
    for name in exp_names:
        sims = []
        for i in range(trials):
            print(i)
            load_file = load_dir + name + str(i) + '.pkl'
            with open(load_file,'rb') as f:
                    if 'ST' in name:
                        pop,_,_,_,_,_,_,_ = pickle.load(f)
                    else:
                        pop,_,_,_,_,_,_ = pickle.load(f)
            sim = get_trainset_similarity(pop)
            sims.append(sim)
        mean_sim = np.mean(sims)
        mean_sims.append(mean_sim)
        std_sim = np.std(sims)
        std_sims.append(std_sim)

        if 'DIF_SF' in name:
            mean_sim_naive = mean_sim
            std_sim_naive = std_sim
    
    for i in range(len(exp_names)):
        name = exp_names[i]
        mean_sim = mean_sims[i]
        std_sim = std_sims[i]
        if 'DIF' in name:
            with open(write_file,'a+') as f:
                f.write(name + '\n')
                f.write('Mean Hausdorffs: {} \n'.format(mean_sim))
                f.write('STD Hausdorffs: {} \n'.format(std_sim))
                f.write('\n')  
        if 'DIT' in name:
            pct_improve = -(mean_sim-mean_sim_naive)/mean_sim_naive
            pval_naive = ttest(mean_sim, std_sim, trials, mean_sim_naive, std_sim_naive, trials)
            with open(write_file,'a+') as f:
                f.write(name + '\n')
                f.write('Mean Hausdorffs: {} \n'.format(mean_sim))
                f.write('STD Hausdorffs: {} \n'.format(std_sim))
                f.write('Mean Percent Improvement: {} \n'.format(pct_improve))
                f.write('Pval Naive: {} \n'.format(pval_naive))
                f.write('\n')

if __name__ == '__main__':
    trials = 5
    load_dir = './pickle/'
    data_key = ['DIF','DIT']
    sparse_key = ['SF','ST']
    lam_array = [0.1,0.2,0.4]
    exp_names = []
    for s in sparse_key:
        if s == 'SF':
            for d in data_key:
                exp_names.append(d + '_' + s + '_')
        else:   
            d = 'DIT'
            for lam in lam_array:
                exp_names.append(d + '_' + s + '_' + str(lam) + '_')

    print(exp_names)
    H1_testing(exp_names,trials,load_dir)
    H23_testing(exp_names,trials,load_dir)
    H4_testing(exp_names,trials,load_dir)
