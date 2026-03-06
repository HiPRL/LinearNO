The official repository of “Transolver is a Linear Transformer: Revisiting Physics-Attention through the Lens of Linear Attention”. [Paper with Appendix](https://arxiv.org/abs/2511.06294)

# Datasets
Please refer to our paper to find the corresponding references for each dataset. The datasets are publicly available and can be downloaded. 

| Dataset       | Link                                                         |
| ------------- | ------------------------------------------------------------ |
| Darcy         | [[Google Cloud]](https://drive.google.com/drive/folders/1UnbQh2WWc6knEHbLn-ZaXrKUZhp7pjt-) |
| NS2d          | [[Google Cloud]](https://drive.google.com/drive/folders/1UnbQh2WWc6knEHbLn-ZaXrKUZhp7pjt-) |
| Airfoil       | [[Google Cloud]](https://drive.google.com/drive/folders/1YBuaoTdOSr_qzaow-G-iwvbUI7fiUzu8) |
| Elasticity    | [[Google Cloud]](https://drive.google.com/drive/folders/1YBuaoTdOSr_qzaow-G-iwvbUI7fiUzu8) |
| Plasticity    | [[Google Cloud]](https://drive.google.com/drive/folders/1YBuaoTdOSr_qzaow-G-iwvbUI7fiUzu8) |
| Pipe          | [[Google Cloud]](https://drive.google.com/drive/folders/1YBuaoTdOSr_qzaow-G-iwvbUI7fiUzu8) | 
| Aifrans       | [[Airfrans]](https://data.isir.upmc.fr/extrality/NeurIPS_2022/Dataset.zip) | 
| ShapeNetCar   | [[ShapeNetCar]](http://www.nobuyuki-umetani.com/publication/mlcfd_data.zip) | 

# Requirements
The required environment and dependencies are listed in the environment.yml file. For example
```
conda env create -f environment.yml
conda activate LinearNO
```

# Running Experiments
To reproduce the results, simply navigate to the corresponding directory of the experiment and run the provided training and testing scripts. For example

```
cd Airfrans
bash scripts/Train.sh
bash scripts/Evaluation.sh
```

## Checkpoint 
For quickly evaluating, you can find the checkpoints at [Google Cloud](https://drive.google.com/drive/folders/10t6g1m89DgDZW7wDa_WYr1EaihmuGPjd?usp=sharing)

# Citation
If you find our work helpful, feel free to give us a cite.

```
@article{hu2025transolver,
  title={Transolver is a Linear Transformer: Revisiting Physics-Attention through the Lens of Linear Attention},
  author={Hu, Wenjie and Liu, Sidun and Qiao, Peng and Sun, Zhenglun and Dou, Yong},
  journal={arXiv preprint arXiv:2511.06294},
  year={2025}
}
```
# Contact 
If you have any questions, please feel free to contact me at: hwjb127@nudt.edu.cn

# Acknowledgement

We appreciate the following github repos a lot for their valuable code base or datasets:

https://github.com/neuraloperator/neuraloperator

https://github.com/neuraloperator/Geo-FNO

https://github.com/thuml/Latent-Spectral-Models

https://github.com/Extrality/AirfRANS

https://github.com/thuml/Transolver

https://github.com/L-I-M-I-T/LatentNeuralOperator

