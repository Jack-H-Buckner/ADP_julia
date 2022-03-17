#!/bin/bash -l
# setting name of job
#SBATCH --job-name=tests

# setting home directory
#SBATCH -D /home/jhbuckne/ADP_julia

# setting standard error output
#SBATCH -e /home/jhbuckne/ADP_julia/Results/stdoutput_%j.txt

# setting standard output
#SBATCH -o /home/jhbuckne/ADP_julia/Results/stdoutput_%j.txt

# setting medium priority
#SBATCH -p med

# setting the max time
#SBATCH -t 00:01:00

# mail alerts at beginning and end of job
# SBATCH --mail-type=BEGIN
# SBATCH --mail-type=END

# send mail here
# SBATCH --mail-user=jhbuckner@ucdavis.edu

# now we'll print out the contents of the R script to the standard output file
cat run.R
echo "ok now for the actual standard output"

module load julia
julia -t 24 run_unknownGrowthRate.jl
