config_name="LinearNO_completer_random_Burgers_ratio02_cycle_Adam"
python prepare.py --data_name Burgers_IC_std
torchrun \
--nnodes 1 \
--nproc_per_node 1 \
--master_port 12349 \
test.py \
--config $config_name \
--device "0" \
--seed 0
