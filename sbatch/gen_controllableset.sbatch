#!/bin/bash
#SBATCH -J generate-controllable-set            # job name
#SBATCH -A gts-kho34							# account name
#SBATCH -N 1 --ntasks-per-node=1     			# amount of nodes, number of tasks per node
#SBATCH -c 32									# number of CPUs per task
#SBATCH --mem=32gb           					# amount of memory per node
#SBATCH -t 48:00:00                   			# wall time (here 2 days)
#SBATCH -q inferno                  			# queue name
#SBATCH -o Report-%j.out           				# report file to be created
#SBATCH --mail-type=BEGIN,END,FAIL      		# event notification, set to email on start, end, or fail
#SBATCH --mail-user=ktomita3@gatech.edu      	# email to send notifications to, ngollins3@gatech.edu


# computations start here
module load anaconda3             # loads python environment (anaconda)
conda activate learn-reach
python gen_controllable_set.py
conda deactivate 

