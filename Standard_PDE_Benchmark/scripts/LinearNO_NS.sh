python exp_ns.py \
--model no_temp \
--gpu 7 \
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
--data_path PDE_dataset/benchmark/fno/NavierStokes_V1e-5_N1200_T20 \


