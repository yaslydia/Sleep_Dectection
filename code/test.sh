#!/bin/bash -l
#SBATCH --job-name=MicroSleep   # Name of job
#SBATCH --account=def-taolu    # adjust this to match the accounting group you are using to submit jobs
#SBATCH --time=0-03:00          # 11 hours
#SBATCH --nodes=1               # 1 node
#SBATCH --ntasks-per-node=15   # Cores per node
#SBATCH --gpus-per-node=a100:1
#SBATCH --mem=32G

module load cuda cudnn scipy-stack


source ~/tensor/bin/activate

pip install /scratch/cxyycl/raybnn_python-0.1.2-cp311-cp311-linux_x86_64.whl

# python /home/cxyycl/scratch/Microsleep-code/code/loadData.py
# python /home/cxyycl/scratch/Microsleep-code/code/CNN_16s/myModel.py
# python /home/cxyycl/scratch/Microsleep-code/code/CNN_16s/train.py
python /home/cxyycl/scratch/Microsleep-code/code/CNN_16s/predict_val.py
# python /home/cxyycl/scratch/Microsleep-code/code/CNN_16s/predict_test.py