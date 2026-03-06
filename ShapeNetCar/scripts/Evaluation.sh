export CUDA_VISIBLE_DEVICES=0

python main_evaluation.py \
--cfd_model=LinearAttnNeuralOperator \
--data_dir ./datas/mlcfd_data/training_data \
--save_dir ./datas/mlcfd_data/preprocessed_data \
