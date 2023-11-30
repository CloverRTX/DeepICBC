# DeepICBC

Synthesizing a safety controller via $k$-inductive control barrier certificates ($k$-ICBC) for controlled discrete-time systems.

## $k$-ICBC
A continuous real-valued function $B(x)$ satisfies the following conditions:
```math
\begin{cases}
&\bigwedge_{0\le i<k}B(f^{i}(\mathbf{x},\mathbf{u}))\le0, &\forall \mathbf{x}\in X_{0}\\[2ex]
&B(\mathbf{x})>0,&\forall \mathbf{x}\in X_{u}\\[2ex]
&\bigwedge_{0\le i<k}B(f^{i}(\mathbf{x},\mathbf{u}))\le0\Longrightarrow B(f^{k}(\mathbf{x},\mathbf{u}))\le0,&\forall \mathbf{x}\in X
\end{cases}
```

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