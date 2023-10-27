#!/bin/bash
#SBATCH -Jreachset_generation                				# job name
#SBATCH -A gts-kho34								# account name
#SBATCH -N 1 --ntasks-per-node=1     			# amount of nodes, number of tasks per node
#SBATCH -c 32									# number of CPUs per task
#SBATCH --mem=32gb           					# amount of memory per node
#SBATCH -t24:00:00                   			# wall time (here 1 days)
#SBATCH -qinferno                  				# queue name
#SBATCH -oReport-%j.out           				# report file to be created
#SBATCH --mail-type=BEGIN,END,FAIL      		# event notification, set to email on start, end, or fail
#SBATCH --mail-user=ktomita3@gatech.edu      	# email to send notifications to, ngollins3@gatech.edu


# computations start here
module load anaconda3/2022.05             # loads python environment (anaconda)
conda activate rahao
python script/gen_reachset.py --n_proc 32 --n_per_proc 1000 --nmax None --nc 30 --icset_path 'out/icset.json' --outdir 'out' --fname 'reachset.json'
conda deactivate 

