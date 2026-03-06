export CUDA_VISIBLE_DEVICES=0

python main.py \
--model LinearAttentionNeuralOperator \
-t full \
--my_path ./datas/Dataset \
--score 1 \
--debug 0 \
--save_name LinearAttentionNeuralOperator