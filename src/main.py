import os, sys
import numpy as np
import matplotlib.pyplot as plt
import copy

import jax
import jax.numpy as jnp
from jax import jit

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

import tqdm as notebook_tqdm
import arviz as az

az.style.use("arviz-darkgrid")

plt.rcParams['font.size'] = 18
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['axes.grid']=False
plt.rcParams['grid.linestyle']='--'
plt.rcParams['grid.linewidth'] = 1.0

np.random.seed(333)

colors = ['red', 'green', 'blue', 'magenta', 'cyan']

numpyro.set_platform("cpu")
numpyro.set_host_device_count(5)

os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=5'

@jit
def gaussian_func(x, h, p, w):
	return h * jnp.power(2.0, -jnp.power((x - p) / w, 2.0))

x = jnp.arange(-1, 1, 0.02)
N = len(x)

K_t = 3
h_t = jnp.array([500, 500, 300])
p_t = jnp.array([-0.25, 0.0, 0.21])
w_t = jnp.array([0.1, 0.1, 0.1])
b_t = 100

batch_peaks = jax.vmap(gaussian_func, in_axes=[0, None, None, None])
peaks = batch_peaks(x, h_t, p_t, w_t)
f_t = jnp.sum(peaks, axis=1) + b_t
y = np.random.poisson(f_t)

plt.figure(figsize=(8, 6))
plt.scatter(x, y, c='gray')
plt.xlabel('Eenergy')
plt.ylabel('Intensity')
plt.savefig('image/data.jpg')
plt.close('all')


beta = 1.0 / jnp.log(N)
print(f"Beta={beta}")

def model(K, x, y):
    h = numpyro.sample( "h", dist.Gamma(concentration=10.0, rate=10.0/jnp.max(y)).expand([K]) )
    p = numpyro.sample( "p", dist.Uniform(low=jnp.min(x), high=jnp.max(x)).expand([K]) )
    w = numpyro.sample( "w", dist.Gamma(concentration=2.0, rate=10.0).expand([K]) )
    
    b = numpyro.sample( "b", dist.Gamma(concentration=50.0, rate=50.0/y[0]) )
    
    batch_peaks = jax.vmap(gaussian_func, in_axes=[0, None, None, None])
    peaks = numpyro.deterministic( "peaks", batch_peaks(x, h, p, w) )
    f = numpyro.deterministic( "f", jnp.sum(peaks, axis=1) + b )
    
    with numpyro.plate("N", len(x)):
        numpyro.sample("y", dist.Poisson(f), obs=y)

def do_annealing(k, beta, x, y):
    log_s_max = int( np.ceil( np.log10( np.max(beta*y) ) ) )
    annealing_scale_list = 10.0**np.arange(0, log_s_max, 1)
    
    for s in annealing_scale_list:
        print(f">> Annealing : scale = {s}")
        scale = s / np.max(y)
        if(s==1):
            kernel = NUTS(model)
        else:
            init_strategy = numpyro.infer.init_to_value(values=init_state)
            kernel = NUTS(model, init_strategy=init_strategy)
        annealing_mcmc = MCMC(kernel, num_warmup=1000, num_samples=1000, num_chains=2)
        annealing_mcmc.run(rng_key=rng_key, K=k, x=x, y=(scale*y))
        samples = annealing_mcmc.get_samples()
        init_state = {
            "h": jnp.mean(samples["h"], axis=0) / scale,
            "p": jnp.mean(samples["p"], axis=0),
            "w": jnp.mean(samples["w"], axis=0),
            "b": jnp.mean(samples["b"]) / scale,
        }
        
    init_state["h"] = beta * ( init_state["h"] / annealing_scale_list[-1] )
    return init_state

if __name__=="__main__":
    rng_key = jax.random.PRNGKey(123)
    k_array = np.arange(2, 6)
    T = { 'H' : beta, 'L' : 1.0 }
    num_warmup = 5000
    num_samples = 5000

    mcmc = {'H' : {}, 'L' : {}}
    for k in k_array:
        print(f"> The number of peaks = {k}")
        init_state = do_annealing(k, beta, x, y)
        
        key = 'H'
        init_strategy = numpyro.infer.init_to_value(values=init_state)
        mcmc[key][k] = MCMC(NUTS(model, init_strategy=init_strategy),
                                                    num_warmup=num_warmup, num_samples=num_samples, num_chains=2)
        mcmc[key][k].run( rng_key=rng_key, K=k, x=x, y=(T[key]*y) )
        
        beta_samples = mcmc[key][k].get_samples()
        init_state = {
            "h": jnp.mean(beta_samples["h"], axis=0) / beta,
            "p": jnp.mean(beta_samples["p"], axis=0),
            "w": jnp.mean(beta_samples["w"], axis=0),
            "b": jnp.mean(beta_samples["b"]) / beta,
        }
        
        key = 'L'
        init_strategy = numpyro.infer.init_to_value(values=init_state)
        mcmc[key][k] = MCMC(NUTS(model, init_strategy=init_strategy),
                                                    num_warmup=num_warmup, num_samples=num_samples, num_chains=2)
        mcmc[key][k].run(rng_key=rng_key, K=k, x=x, y=(T[key]*y))

    wbic_array = np.array([])
    bic_array = np.array([])
    for k in k_array:
        # log_likelihood = numpyro.infer.log_likelihood(model=model, posterior_samples=mcmc['H'][k].get_samples(), K=k, x=x, y=(T['H']*y))
        samples_H = copy.deepcopy( mcmc['H'][k].get_samples() )
        samples_H['h'] = samples_H['h'] / beta
        samples_H['peaks'] = samples_H['peaks'] / beta
        samples_H['f'] = samples_H['f'] / beta
        log_likelihood = numpyro.infer.log_likelihood(model=model, posterior_samples=samples_H, K=k, x=x, y=(T['L']*y))
        wbic = -1.0 * float( jnp.mean( np.sum(log_likelihood['y'], axis=1) ) )
        wbic_array = np.append(wbic_array, wbic)
        
        log_likelihood = numpyro.infer.log_likelihood(model=model, posterior_samples=mcmc['L'][k].get_samples(), K=k, x=x, y=(T['L']*y))
        d = k*3+1
        bic = np.min(-1.0*np.sum(log_likelihood['y'], axis=1)) + (d/2)*np.log(N)
        bic_array = np.append(bic_array, bic)
        
        print(f"K={k}, bic={bic:.2f}, wbic={wbic:.2f}")
    
    F = wbic_array - np.min(wbic_array)
    prob = np.exp( -F ) / np.sum( np.exp( -F ) )

    fig = plt.figure(figsize=(8, 6))
    ax1 = plt.subplot(111)
    plt.ylabel('Probablity [%]')
    plt.xlabel(r'The Number of peaks $K$')
    ax1.bar(k_array,  100.0 * prob, color='gray')
    plt.ylim(0, 100)

    ax2 = ax1.twinx()
    ax2.plot(k_array, bic_array, marker='o', color='c', markersize=6, label='BIC')
    ax2.plot(k_array, wbic_array, marker='o', color='b', markersize=8, label='WBIC')
    plt.legend(fontsize=14)
    plt.ylabel('WBIC \ BIC')
    plt.savefig('image/wbic.jpg')
    plt.close('all')


    for k in k_array:
        plt.figure(figsize=(8, 6))
        key = 'L'
        plt.scatter(x, T[key]*y)
        for i in range(100):
            plt.plot(x, mcmc[key][k].get_samples()['f'][-i, :], alpha=0.05, color='gray')
            for k_i in range(k):
                plt.plot(x, mcmc[key][k].get_samples()['peaks'][-i, :, k_i], alpha=0.05, color=colors[k_i])
        plt.xlabel('Energy')
        plt.ylabel('Frequency')
        plt.savefig(f'image/fitting_{key}_{k:03}.jpg')
