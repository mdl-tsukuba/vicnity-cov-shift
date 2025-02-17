import numpy as np
import scipy.stats
import sklearn.neighbors
import matplotlib.pyplot as plt
import math
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
import pickle
import seaborn as sns
import pandas as pd
import argparse 

np.random.seed(0)

def oracle_choise_k_old(np,nq,tau,phi,alpha,beta):
    return math.floor((np**(1/(2+beta+tau/alpha))+nq**(1/(2+beta+phi/alpha)))**2)

def oracle_choise_k_ours(np,nq,tau,phi,alpha,beta):
    return math.floor((np**(1/(2+beta+tau/alpha))+nq**(1/(2+beta+phi/alpha)))**2)

def eta(x,qx_scale=1,alpha = 1):
    if x > 0:
        return pow(x,alpha)/2 + 1/2 
    else:
        return -pow(abs(x),alpha)/2 + 1/2

def bayes_classifier(eta,x,qx_scale,alpha):
    return 1 if eta(x,qx_scale,alpha) >= 1/2 else  0

def upperbound_old(np,nq,tau,phi,alpha,beta):
    if alpha == 1: 
        return (np**((1+beta)/(2 + beta + tau/alpha))+nq**((1+beta)/(2 + beta + phi/alpha)))**(-1)  * math.log(2*(np+nq))
    else:
        return (np**((1+beta)/(2 + beta + tau/alpha))+nq**((1+beta)/(2 + beta + phi/alpha)))**(-1) 
    
def upperbound_ours(np,nq,tau,phi,alpha,beta):
    if alpha == 1: 
        return (np**((1+beta)/(2 + beta + tau/alpha))+nq**((1+beta)/(2 + beta + phi/alpha)))**(-1)  * math.log(2*(np+nq))
    else:
        return (np**((1+beta)/(2 + beta + tau/alpha))+nq**((1+beta)/(2 + beta + phi/alpha)))**(-1) 

def Xp_rvs(px_scale,qx_scale,max_size,gamma = 2):
    return scipy.stats.beta.rvs(a=(gamma + 2) /2,b = (gamma + 2 )/2,loc = - px_scale/2, scale=px_scale,size=max_size)
    
def excess_error_by_eta(X,Y,eta,qx_scale,alpha):
    return 2 * np.mean(np.array([abs(eta(x,qx_scale,alpha) - 1/2) * (1 if Y[i] != bayes_classifier(eta,x,qx_scale,alpha) else 0) for i,x in enumerate(X)]))

def sample_error_ex(tau,phi,alpha,tau_old,phi_old,px_scale,qx_scale,max_size,test_size,q_size,sample_size_range):

    excess_error_list_old = []
    excess_error_list_ours = []

    Xp = Xp_rvs(px_scale,qx_scale,max_size,gamma=tau)
    Yp = np.array([scipy.stats.binom.rvs(1, eta(x, qx_scale,alpha), size=1)[0] for x in Xp])
    
    Xq = scipy.stats.uniform.rvs(loc = -qx_scale/ 2, scale=qx_scale, size=q_size)
    Yq = np.array([scipy.stats.binom.rvs(1, eta(x,qx_scale,alpha), size=1)[0] for x in Xq])

    Xqtest = scipy.stats.uniform.rvs(loc = -qx_scale/2, scale=qx_scale, size=test_size)
    Yqtest = np.array([scipy.stats.binom.rvs(1, eta(x,qx_scale,alpha), size=1)[0] for x in Xqtest])

    for sample_size in sample_size_range:
        k_old = oracle_choise_k_old(sample_size,q_size,tau_old,phi_old,alpha,beta)
        k_ours = oracle_choise_k_ours(sample_size,q_size,tau,phi,alpha,beta)

        knn_old = sklearn.neighbors.KNeighborsClassifier(n_neighbors=k_old)
        knn_ours = sklearn.neighbors.KNeighborsClassifier(n_neighbors=k_ours)

        knn_old.fit(np.concatenate([Xp[:sample_size],Xq[:q_size]]).reshape(-1, 1) ,np.concatenate([Yp[:sample_size],Yq[:q_size]]))
        knn_ours.fit(np.concatenate([Xp[:sample_size],Xq[:q_size]]).reshape(-1, 1) ,np.concatenate([Yp[:sample_size],Yq[:q_size]]))

        old_predicted = knn_old.predict(Xqtest.reshape(-1,1))
        ours_predicted = knn_ours.predict(Xqtest.reshape(-1,1))

        excess_error_list_old.append(excess_error_by_eta(Xqtest,old_predicted,eta,qx_scale,alpha))
        excess_error_list_ours.append(excess_error_by_eta(Xqtest,ours_predicted,eta,qx_scale,alpha))

        print(sample_size,excess_error_by_eta(Xqtest,ours_predicted,eta,qx_scale,alpha))

    result = {'ee_old':[],'ee_ours':[]}
    result['ee_old'] = np.array(excess_error_list_old)
    result['ee_ours'] = np.array(excess_error_list_ours)

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha',type=float)
    parser.add_argument('--tau',type=float)
    parser.add_argument('--plotonly', action='store_true')
    parser.add_argument('--fontsize', default=16)
    parser.add_argument('--tick_fontsize',default=14)
    args = parser.parse_args()
    plot_only = args.plotonly
    fontsize = args.fontsize
    tick_fontsize = args.tick_fontsize

    ex_num = 10

    tau = args.tau
    phi = 1
    tau_old = np.inf
    phi_old = 1
    alpha= args.alpha
    beta = 1/alpha

    qx_scale = 2
    px_scale = qx_scale *  ((pow(8,1/alpha) * 2-1)/ (pow(8,1/alpha) * 2)) 

    q_size = 10
    start_size = 2**8
    max_size = 2**18
    sample_size_step = 2
    test_size = 5000

    sample_size_range = [start_size]
    for i in range(math.ceil(math.log(max_size,sample_size_step)) - math.ceil(math.log(start_size,sample_size_step))):
        sample_size_range.append(sample_size_range[-1] * sample_size_step)

    if plot_only:
        with open(f'newsim-sample-excess-e{ex_num}-loglog-s{start_size}p{max_size}q{q_size}-simple-tau{tau}-alpha{alpha}-qsc{qx_scale}.bin', 'br') as p:
            ee_and_ub= pickle.load(p)
            results = ee_and_ub['ee']
            upperbound_list_old = ee_and_ub['ub'][0]
            upperbound_list_ours = ee_and_ub['ub'][1]
    else:
        upperbound_list_old = []
        upperbound_list_ours = []

        for sample_size in sample_size_range:
            upperbound_list_old.append(upperbound_old(sample_size,q_size,tau_old,phi_old,alpha,beta))
            upperbound_list_ours.append(upperbound_ours(sample_size,q_size,tau,phi,alpha,beta))

        results = {'ee_old':[],'ee_ours':[]}
        for i in range(ex_num):
            result = sample_error_ex(tau,phi,alpha,tau_old,phi_old,px_scale,qx_scale,max_size,test_size,q_size,sample_size_range)
            for key in result:
                results[key].append(result[key])

        for key in results:
            results[key] = np.array(results[key])

        excess_error_list_old = np.average(results['ee_old'],axis=0)
        excess_error_list_ours = np.average(results['ee_ours'],axis=0)

        upperbound_list_old = np.array(upperbound_list_old)  * (excess_error_list_old[0] / upperbound_list_old[0])
        upperbound_list_ours = np.array(upperbound_list_ours) * (excess_error_list_ours[0] / upperbound_list_ours[0])

        with open(f'newsim-sample-excess-e{ex_num}-loglog-s{start_size}p{max_size}q{q_size}-simple-tau{tau}-alpha{alpha}-qsc{qx_scale}.bin', 'wb') as p:
            pickle.dump({'ee':results,'ub':[upperbound_list_old,upperbound_list_ours]}, p)

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['text.usetex'] = True
    plt.rcParams['text.latex.preamble'] = '\\usepackage[T1]{fontenc}\n\\usepackage{lmodern}'
    plt.rcParams["font.size"] = fontsize
    figure = plt.figure()
    ax = figure.add_subplot()

    sns.lineplot(pd.DataFrame(np.array([np.tile(sample_size_range,ex_num),np.array(results['ee_old']).reshape(-1)]).T,columns=['$n_P$','excess error']),x='$n_P$',y='excess error',errorbar=('pi',50),label='k-NN (PMW)')
    sns.lineplot(pd.DataFrame(np.array([np.tile(sample_size_range,ex_num),np.array(results['ee_ours']).reshape(-1)]).T,columns=['$n_P$','excess error']),x='$n_P$',y='excess error',errorbar=('pi',50),label='k-NN (our)')

    ax.plot(sample_size_range, upperbound_list_old,label='Bound (PMW)')
    ax.plot(sample_size_range, upperbound_list_ours,label='Bound (our)')

    ax.set_yscale('log',base=2)
    ax.set_xscale('log',base=2)
    ax.legend(handlelength=0.6, borderpad=0.2, labelspacing=0.1, handletextpad=0.2, borderaxespad=0.2)

    plt.tick_params(labelsize=tick_fontsize)
    plt.yticks([pow(2,-i) for i in range(5,14) ], ['$2^{'+f'{-i}'+'}$' for i in range(5,14)])
    plt.tight_layout()
    plt.savefig(f'newsim-sample-excess-e{ex_num}-loglog-s{start_size}p{max_size}q{q_size}-simple-tau{tau}-alpha{alpha}-qsc{qx_scale}.pdf')


