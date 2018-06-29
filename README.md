# GenOpt: Genetic Hyperparameter Tuner

*Created by Bhairav Chidambaram*  

### Requirements

Python 3 (that's it)

### Usage

GenOpt defines three classes representing types of variables  
- Continuous (e.g. learning rate: [0.0001 - 0.1])
- Discrete (e.g. number of layers: [2, 3, 4, 5])
- Categorical (e.g. activation: [sigmoid, tanh, relu])

Each class holds the range or set of values that variable can take.
Both the Continuous and Discrete variables store the min and max value.
The Categorical variable holds a list of values.
Together, the set of variables for a model determine the hyperparameter search space.
See "example.py" to see how these variables are initialized.  

Once you've defined the set of variables, you will need to create an evaluation function.
Given a tuple of hyperparameters (equal in length to the number of variables), this function should return a single number: the validation loss.
Using tuple-unpacking syntax, it is easy to extract the individual values from this tuple and use them to initialize your model inside the evaluation function.
To run GenOpt, simply call the optimize function with this evaluation function and the list of variables as input.
GenOpt will print the results of hyperparameter tuning to the terminal.
Again, see "example.py" to see how all this is done in code.

### Etc.

Some metaparameters for the genetic algorithm have also been set. 
See GenOpt.py for details.
By default, GenOpt is single-threaded, however if you have multiple cores you can set threads = n > 1 to evaluate your models in parallel and speed up tuning.
