## Tuning network parameters through a genetic algorithm

This fork is used to explore how to tune the neural network parameters with a genetic algorithm, in order to get the better training time and network performance. The original code of the neural network can be found here:
https://github.com/mnielsen/neural-networks-and-deep-learning

To run the genetic algorithm simply use

`python genetic.py`

you can also supply optional arguments to tune the parameters of the algorithm

```
--popsize <population size (integer)>
--gen <number of generations (integer)>
--cross <number of crossovers per generation (integer)>
--mut_prob <mutation probability (floating point number)>
```
