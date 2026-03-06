config_name="LinearNO_completer_random_Burgers_ratio0005_cycle_Adam"
python prepare.py --data_name Burgers_IC_std
torchrun \
--nnodes 1 \
--nproc_per_node 1 \
--master_port 12349 \
train.py \
--config $config_name \
--device "0" \
--seed 0
