# Anomaly Detection in Aeronautics Data with Quantum-compatible Discrete Deep Generative Model


Code for [Anomaly Detection in Aeronautics Data with Quantum-compatible Discrete Deep Generative Model] (https://iopscience.iop.org/article/10.1088/2632-2153/ace756; https://arxiv.org/abs/2303.12302).


## Requirements:
```
python=3.9.13
pytorch=1.13.1
numpy=1.21.5
scipy=1.9.1
scikit-learn=1.0.2
matplotlib=3.5.2
seaborn=0.11.2
kmodes=0.12.2
inflect=6.0.4
```


## Training:
```
discrete_stochastic_training.py (-md validation -m boltzmann -e 400 ...)
```


## Evaluation:
```
discrete_stochastic_evaluation.py (-md validation -m boltzmann -e 400 ...)
```


## Example
```
The Jupyter notebook file dvae_example.ipynb contains guidance on how to train and evaluate our DVAE model for the purpose of anomaly detection in time-series data.
```


## Citation
```
@article{Templin_2023,
  title={Anomaly detection in aeronautics data with quantum-compatible discrete deep generative model},
  author={Templin, Thomas and Memarzadeh, Milad and Vinci, Walter and Lott, P. Aaron and Akbari Asanjan, Ata and Alexiades Armenakas, Anthony and Rieffel, Eleanor},
  journal={Machine Learning: Science and Technology},
  volume={4},
  number={3},
  pages={035018},
  year={2023},
  publisher={IOP Publishing}
}
@article{templin2023anomaly,
  title={Anomaly detection in aeronautics data with quantum-compatible discrete deep generative model},
  author={Templin, Thomas and Memarzadeh, Milad and Vinci, Walter and Lott, P. Aaron and Akbari Asanjan, Ata and Alexiades Armenakas, Anthony and Rieffel, Eleanor},
  journal={arXiv preprint arXiv:2303.12302},
  year={2023}
}
```
