## Tuning network parameters through a genetic algorithm

This fork is used to explore how to tune the neural network parameters with a genetic algorithm, in order to get the better training time and network performance. The original code of the neural network can be found here:
https://github.com/mnielsen/neural-networks-and-deep-learning

To run the genetic algorithm simply use

`python genetic.py` for a single layer neural network

or

`python genetic_conv.py` for a convolutional neural network

you can also supply optional arguments to tune the parameters of the algorithm

```
--popsize <population size (integer)>
--gen <number of generations (integer)>
--cross <number of crossovers per generation (integer)>
--mut_prob <mutation probability (floating point number)>
```
 a file named `report.txt` containing the results of the run will be generated in the `report` directory.
 
 More details about the implementation and the approach in general can be found in report.pdf

## Requirements

python: https://www.python.org/

numpy: http://www.numpy.org/ (if pip is installed, simply type `pip install numpy`)

tensorflow: https://www.tensorflow.org/ (if pip is installed, simply type `pip install tensorflow`)

keras: https://keras.io/ (if pip is installed, simply type `pip install keras`)
