python exp_airfoil.py \
--model conv_temp \
--gpu 0 \
--n-hidden 128 \
--n-heads 8 \
--n-layers 8 \
--lr 0.001 \
--weight_decay 0.00001 \
--max_grad_norm 1 \
--downsamplex 1 \
--downsampley 1 \
--batch-size 4 \
--key_ratio 64 \
--mlp_ratio 1 \
--unified_pos 0 \
--ref 8 \
--eval 1 \
--save_name  airfoil_LinearAttnNeuralOperator \
--data_path /home/ubuntu/shared-data/PDE_dataset/benchmark/airfoil/naca 

python exp_darcy.py \
--model conv_temp \
--gpu 0 \
--n-hidden 128 \
--n-heads 8 \
--n-layers  8 \
--lr 0.001 \
--max_grad_norm 1 \
--weight_decay 0.000001 \
--batch-size 4 \
--key_ratio 64 \
--unified_pos 0 \
--mlp_ratio 1 \
--ref 8 \
--eval 1 \
--downsample 5 \
--save_name darcy_LinearAttnNeuralOperator \
--data_path /home/ubuntu/shared-data/PDE_dataset/benchmark/fno 

python exp_elas.py \
--model temp \
--gpu 0 \
--n-hidden 128 \
--n-heads 8 \
--n-layers 8 \
--lr 0.001 \
--max_grad_norm 1 \
--batch-size 1 \
--key_ratio 64 \
--unified_pos 0 \
--ref 8 \
--eval 1 \
--save_name elas_LinearAttnNeuralOperator \
--data_path /home/ubuntu/shared-data/PDE_dataset/benchmark/ 

python exp_ns.py \
--model no_temp \
--gpu 0 \
--n-hidden 256 \
--n-heads 8 \
--n-layers 8 \
--lr 0.001 \
--weight_decay 0.000001 \
--batch-size 2 \
--key_ratio 32 \
--unified_pos 1 \
--mlp_ratio 2 \
--ref 10 \
--eval 1 \
--save_name ns_LinearAttnNeuralOperator \
--data_path /home/ubuntu/shared-data/PDE_dataset/benchmark/fno/NavierStokes_V1e-5_N1200_T20 

python exp_pipe.py \
--model conv_temp \
--gpu 0 \
--n-hidden 128 \
--n-heads 8 \
--n-layers 8 \
--mlp_ratio 1 \
--lr 0.001 \
--weight_decay 0.00001 \
--max_grad_norm 1 \
--batch-size 4   \
--key_ratio 64 \
--unified_pos 0 \
--ref 8 \
--eval  1 \
--save_name pipe_LinearAttnNeuralOperator \
--data_path /home/ubuntu/shared-data/PDE_dataset/benchmark/pipe 

python exp_plas.py \
--gpu 0 \
--model conv \
--n-hidden 128 \
--n-heads 8 \
--n-layers 8 \
--lr 0.001 \
--max_grad_norm 1 \
--weight_decay 0.000001 \
--batch-size 8 \
--key_ratio 64 \
--mlp_ratio 1 \
--unified_pos 0 \
--ref 8 \
--eval 1 \
--save_name plas_LinearAttnNeuralOperator \
--data_path /home/ubuntu/shared-data/PDE_dataset/benchmark/plasticity 
