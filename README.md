# Quadrotor Velocity Control with DDPG
This repository provides the code for implementing the baseline DDPG-based quadrotor velocity control algoirthm in paper [[Deterministic policy gradient with integral compensator for robust quadrotor control](https://ieeexplore.ieee.org/abstract/document/8600717/)].

**Note:** This code repository originates from a research project around 2017. As it has not been updated since then, it is now shared solely for learning and educational purposes. Please be aware that this repository will not receive any further maintenance or updates.

## Requirements

- Python 2.x, 3.x
- Tensorflow 1.x
- Jupyter Notebook

## Code Structure

- `DDPG_Agent.py`: Implementation of the DDPG algorithm.
- `QuadModel05.py`: Contains the simplified dynamic model of the Quadrotor.
- `Train_DDPG_baseline.ipynb`: Main script to initialize and run the baseline DDPG algorithm on the quadrotor model. 

## Usage
To train the DDPG controller, use Jupyter notebook to run  `Train_DDPG_baseline.ipynb`.

## Paper Citation

```bibtex
@article{wang2019deterministic,
  title={Deterministic policy gradient with integral compensator for robust quadrotor control},
  author={Wang, Yuanda and Sun, Jia and He, Haibo and Sun, Changyin},
  journal={IEEE Transactions on Systems, Man, and Cybernetics: Systems},
  volume={50},
  number={10},
  pages={3713--3725},
  year={2019},
  publisher={IEEE}
}
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.



Yuanda Wang

Nov. 12, 2024