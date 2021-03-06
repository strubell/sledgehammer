timestamp=`date +%Y-%m-%d-%H-%M-%S`
PROJ_ROOT=/private/home/strubell/research/sledgehammer
JOBSCRIPTS=slurm_scripts
mkdir -p ${JOBSCRIPTS}
queue=dev
#queue=learnfair
MEM="8g"
SAVE_ROOT=$PROJ_ROOT/slurm-$timestamp

working_dir="/private/home/strubell/research/sledgehammer/models"
data_dir="/private/home/strubell/research/sledgehammer/data_dir"
dataset="SST-2"

#layers="0_3_5_11"
#bert_model="bert-base-uncased"
layers="0_4_12_23"
bert_model="bert-large-uncased"

for jobid in $( seq 0 9 ); do
    cname="${dataset}_${bert_model}_${jobid}"
    SAVE="${SAVE_ROOT}/${cname}"
    mkdir -p ${SAVE}
    SCRIPT=${JOBSCRIPTS}/run.${cname}.sh
    SLURM=${JOBSCRIPTS}/run.${cname}.slrm
    echo "#!/bin/sh" > ${SCRIPT}
    echo "#!/bin/sh" > ${SLURM}
    echo "#SBATCH --job-name=$cname" >> ${SLURM}
    echo "#SBATCH --output=$SAVE/train.log" >> ${SLURM}
    echo "#SBATCH --error=$SAVE/train.err" >> ${SLURM}
    echo "#SBATCH --signal=USR1@120" >> ${SLURM}
    echo "#SBATCH --partition=${queue}" >> ${SLURM}
    echo "#SBATCH --mem=$MEM" >> ${SLURM}
    echo "#SBATCH --gres=gpu:1" >> ${SLURM}
    echo "#SBATCH --time=12:00:00" >> ${SLURM}
    if [[ $cname =~ "large" ]]; then
        echo "#SBATCH --constraint=volta32gb" >> ${SLURM}
    fi
    echo "srun sh ${SCRIPT}" >> ${SLURM}
    echo "echo $cname " >> ${SCRIPT}
    echo "cd $PROJ_ROOT" >> ${SCRIPT}
    echo "python scripts/train_model.py \\
          -t $bert_model \\
          -l $layers \\
          --data_dir $data_dir \\
          -d $dataset \\
          -w $SAVE" \
    >> ${SCRIPT}
    echo "Writing output: $SAVE/train.log"
    sbatch ${SLURM}
done
