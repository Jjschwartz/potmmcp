#!/bin/bash

# This is a script for launching tensorboard on the cluster.
# This will launch tensorboard on a node in the cluster and serve to port 6006
# (the default port), or closest unused port
#
# Tensorboard can then be accessed on your local machine by running:
# ssh -L 16006:127.0.0.1:6006 ANUID@NODENAME -J ANUID@cluster1.cecs.anu.edu.au
#
# or more precisely for me specifically (since sococluster is in my ssh config):
# ssh -L 16006:127.0.0.1:6006 u1078789@NODENAME -J sococluster
#
# Where
#   6006 may need to be replaced by a different port
#   NODENAME is the name of the node that this job gets run (xcs-vnode-1)
#
# The ssh command essentially creates a SSH tunnel from the local host to the
# node that the job is running on, throught the cluster login node.
# It then forwards all traffic on port 6006 on the cluster job node to port
# 16006 on the local host.
#
# Tensorboard can then be accessed at: http://localhost:16006/

#!/bin/bash
#SBATCH --job-name=tensorboard        # Job name
#SBATCH --ntasks=1                    # Run on a single CPU
#SBATCH --mem=8GB                     # Job memory request
#SBATCH --time=03:00:00               # Time limit hrs:min:sec
#SBATCH --partition=normal
#SBATCH --nodelist=xcs-vnode-1
#SBATCH --output=tensorboard%j.log    # Standard output and error log

# Start Tensorbxoard port
export PORT=6006

testport() {
nc -z 127.0.0.1 $1
return $?
}

echo "---- Allocated ----"
hostname; date
while testport $PORT
do
echo "Port $PORT in use. Trying next port..."
((PORT=PORT+1))
sleep 0.5
done

echo "-- Forward port from local host --"
echo "ssh -L 16006:localhost:$PORT $HOSTNAME -J cluster1.cecs.anu.edu.au"
echo "-- Starting Tensorboard --"
singularity exec ~/containers/baposgmcp.sif \
			tensorboard --logdir ~/ray_results --port $PORT
echo "---- Completed ----"
