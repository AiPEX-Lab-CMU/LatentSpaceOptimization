import argparse
import run_evo as re
import os, shutil
from pytictoc import TicToc
from datetime import timedelta

'''
Todo:
-use same set of initial designs for each trial
'''

def run_multi(lam_array,num_params=1024,maxiter=50,trials=5):
	tictoc = TicToc()
	tictoc.tic() 
	for i in range(trials):
		out_name = 'DIF_SF_' + str(i)
		print('Run:' + out_name)
		re.run_evo(data_init=False,sparse=False,out_name=out_name,num_params=num_params,maxiter=maxiter)

		out_name = 'DIT_SF_' + str(i)
		print('Run:' + out_name)
		re.run_evo(data_init=True,sparse=False,out_name=out_name,num_params=num_params,maxiter=maxiter)

		for lam in lam_array:
				norm = 0
				# out_name = 'DIF_ST_' + str(lam) +  '_' + str(i)
				# print('Run:' + out_name)
				# re.run_evo(data_init=False,sparse=True,lam=lam,out_name=out_name,num_params=num_params,maxiter=maxiter)

				out_name = 'DIT_ST_' + str(lam) + '_' + str(i)
				print('Run:' + out_name)
				re.run_evo(data_init=True,sparse=True,lam=lam,out_name=out_name,num_params=num_params,maxiter=maxiter,norm=norm)

	elapsed = timedelta(seconds = tictoc.tocvalue())
	print('Total Execution Time: {}'.format(elapsed))

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--lam_array', type=float,nargs = '+', help='lambda values to evaluate')
	parser.add_argument('--num_params', type=int, default = 1024,  help='size of latent space')
	parser.add_argument('--maxiter', type=int, default = 35,  help='maximum number of iterations')
	parser.add_argument('--trials', type=int, default = 5,  help='maximum number of iterations')
	args = parser.parse_args()
	print(args)

	run_multi(args.lam_array,args.num_params,args.maxiter,args.trials)