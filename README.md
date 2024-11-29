# DeepICBC

Synthesizing a safety controller via $k$-inductive control barrier certificates ($k$-ICBC) for controlled discrete-time systems.


## Controlled discrete-time systems

```math
\mathbf{x}_{t+1} = f(\mathbf{x}_{t},\mathbf{u}_{t})
```
Under a safety controller $\mathbf{u}_t = u(\mathbf{x})$, system trajectories remain in the safe area.

### Prerequisites
Python 3.7+ and PyTorch 1.11+ are recommended.\
Here are some required Python packages:
```
numpy
pandas
sympy
gurobipy
```

## Running the benchmarks
To start synthesis:
```
python k-ICBC ex1/main.py
```
### Tips
Some existing results are saved in Pre_Results.
```
k-ICBC ex1/NN_Train_Result/Pre_Result
```
If our work helps you, please kindly cite our paper:
```
@ARTICLE{10538169,
  author={Ren, Tianxiang and Lin, Wang and Ding, Zuohua},
  journal={IEEE Transactions on Reliability}, 
  title={Formal Synthesis of Safety Controllers via $k$-Inductive Control Barrier Certificates}, 
  year={2024},
  volume={},
  number={},
  pages={1-10},
  keywords={Safety;Control systems;Discrete-time systems;Training;Polynomials;Neural networks;Supervised learning;  $k$  -inductive control barrier certificates ( $k$ -ICBCs);discrete-time dynamical systems;formal verification;mixed integer linear programs;safety controllers},
  doi={10.1109/TR.2024.3399739}}
```


## Authors

* **Tianxiang Ren** - student @ ZSTU

<!--
## Copyright notice:

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc
-->