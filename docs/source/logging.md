# Logging and the Lifecycle Summary

This page covers how to see what PyGAD is doing: printing a lifecycle summary and logging the outputs.

## Print Lifecycle Summary

In [PyGAD 2.19.0](https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-19-0), a new method called `summary()` is supported. It prints a Keras-like summary of the PyGAD lifecycle showing the steps, callback functions, parameters, etc.

 This method accepts the following parameters:

- `line_length=70`: An integer representing the length of the single line in characters.
- `fill_character=" "`: A character to fill the lines.
- `line_character="-"`: A character for creating a line separator.
- `line_character2="="`: A secondary character to create a line separator.
- `columns_equal_len=False`: The table rows are split into equal-sized columns or split subjective to the width needed.
- `print_step_parameters=True`: Whether to print extra parameters about each step inside the step. If `print_step_parameters=False` and `print_parameters_summary=True`, then the parameters of each step are printed at the end of the table.
- `print_parameters_summary=True`: Whether to print parameters summary at the end of the table. If `print_step_parameters=False`, then the parameters of each step are printed at the end of the table too.

Here is a quick example.

```python
import pygad
import numpy

function_inputs = [4,-2,3.5,5,-11,-4.7]
desired_output = 44

def genetic_fitness(solution, solution_idx):
    output = numpy.sum(solution*function_inputs)
    fitness = 1.0 / (numpy.abs(output - desired_output) + 0.000001)
    return fitness

def on_gen(ga):
    pass

def on_crossover_callback(a, b):
    pass

ga_instance = pygad.GA(num_generations=100,
                       num_parents_mating=10,
                       sol_per_pop=20,
                       num_genes=len(function_inputs),
                       on_crossover=on_crossover_callback,
                       on_generation=on_gen,
                       parallel_processing=2,
                       stop_criteria="reach_10",
                       fitness_batch_size=4,
                       crossover_probability=0.4,
                       fitness_func=genetic_fitness)
```

Then call the `summary()` method to print the summary with the default parameters. Note that entries for the crossover and generation callbacks are created because they are implemented through `on_crossover_callback()` and `on_gen()`, respectively.

```python
ga_instance.summary()
```

```bash
----------------------------------------------------------------------
                           PyGAD Lifecycle                           
======================================================================
Step                   Handler                            Output Shape
======================================================================
Fitness Function       genetic_fitness()                  (1)      
Fitness batch size: 4
----------------------------------------------------------------------
Parent Selection       steady_state_selection()           (10, 6)  
Number of Parents: 10
----------------------------------------------------------------------
Crossover              single_point_crossover()           (10, 6)  
Crossover probability: 0.4
----------------------------------------------------------------------
On Crossover           on_crossover_callback()            None     
----------------------------------------------------------------------
Mutation               random_mutation()                  (10, 6)  
Mutation Genes: 1
Random Mutation Range: (-1.0, 1.0)
Mutation by Replacement: False
Allow Duplicated Genes: True
----------------------------------------------------------------------
On Generation          on_gen()                           None     
Stop Criteria: [['reach', 10.0]]
----------------------------------------------------------------------
======================================================================
Population Size: (20, 6)
Number of Generations: 100
Initial Population Range: (-4, 4)
Keep Elitism: 1
Gene DType: [<class 'float'>, None]
Parallel Processing: ['thread', 2]
Save Best Solutions: False
Save Solutions: False
======================================================================
```

We can set the `print_step_parameters` and `print_parameters_summary` parameters to `False` to not print the parameters.

```python
ga_instance.summary(print_step_parameters=False,
                    print_parameters_summary=False)
```

```bash
----------------------------------------------------------------------
                           PyGAD Lifecycle                           
======================================================================
Step                   Handler                            Output Shape
======================================================================
Fitness Function       genetic_fitness()                  (1)      
----------------------------------------------------------------------
Parent Selection       steady_state_selection()           (10, 6)  
----------------------------------------------------------------------
Crossover              single_point_crossover()           (10, 6)  
----------------------------------------------------------------------
On Crossover           on_crossover_callback()            None     
----------------------------------------------------------------------
Mutation               random_mutation()                  (10, 6)  
----------------------------------------------------------------------
On Generation          on_gen()                           None     
----------------------------------------------------------------------
======================================================================
```

## Logging Outputs

In [PyGAD 3.0.0](https://pygad.readthedocs.io/en/latest/releases.html#pygad-3-0-0), the `print()` statement is no longer used and the outputs are printed using the [logging](https://docs.python.org/3/library/logging.html) module. A new parameter called `logger` is supported to accept a user-defined logger.

```python
import logging

logger = ...

ga_instance = pygad.GA(...,
                       logger=logger,
                       ...)
```

The default value for this parameter is `None`. If there is no logger passed (i.e. `logger=None`), then a default logger is created to log the messages to the console exactly like how the `print()` statement works.

Some advantages of using the [logging](https://docs.python.org/3/library/logging.html) module instead of the `print()` statement are:

1. The user has more control over the printed messages, especially in a project that uses multiple modules where each module prints its messages. A logger can organize the outputs.
2. Using the proper `Handler`, the user can log the output messages to files, not only to the console. So, it is much easier to record the outputs.
3. The format of the printed messages can be changed by customizing the `Formatter` assigned to the Logger.

This section gives some quick examples to use the `logging` module and then gives an example to use the logger with PyGAD.

### Logging to the Console

This is an example to create a logger to log the messages to the console.

```python
import logging

# Create a logger
logger = logging.getLogger(__name__)

# Set the logger level to debug so that all the messages are printed.
logger.setLevel(logging.DEBUG)

# Create a stream handler to log the messages to the console.
stream_handler = logging.StreamHandler()

# Set the handler level to debug.
stream_handler.setLevel(logging.DEBUG)

# Create a formatter
formatter = logging.Formatter('%(message)s')

# Add the formatter to handler.
stream_handler.setFormatter(formatter)

# Add the stream handler to the logger
logger.addHandler(stream_handler)
```

Now, we can log messages to the console with the format specified in the `Formatter`.

```python
logger.debug('Debug message.')
logger.info('Info message.')
logger.warning('Warn message.')
logger.error('Error message.')
logger.critical('Critical message.')
```

The outputs are identical to those returned using the `print()` statement.

```
Debug message.
Info message.
Warn message.
Error message.
Critical message.
```

By changing the format of the output messages, we can have more information about each message.

```python
formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
```

This is a sample output.

```python
2023-04-03 18:46:27 DEBUG: Debug message.
2023-04-03 18:46:27 INFO: Info message.
2023-04-03 18:46:27 WARNING: Warn message.
2023-04-03 18:46:27 ERROR: Error message.
2023-04-03 18:46:27 CRITICAL: Critical message.
```

Note that you may need to clear the handlers after finishing the execution. This is to make sure no cached handlers are used in the next run. If the cached handlers are not cleared, then the single output message may be repeated. 

```python
logger.handlers.clear()
```

### Logging to a File

This is another example to log the messages to a file named `logfile.txt`. The formatter prints the following about each message:

1. The date and time at which the message is logged.
2. The log level.
3. The message.
4. The path of the file.
5. The line number of the log message.

```python
import logging

level = logging.DEBUG
name = 'logfile.txt'

logger = logging.getLogger(name)
logger.setLevel(level)

file_handler = logging.FileHandler(name, 'a+', 'utf-8')
file_handler.setLevel(logging.DEBUG)
file_format = logging.Formatter('%(asctime)s %(levelname)s: %(message)s - %(pathname)s:%(lineno)d', datefmt='%Y-%m-%d %H:%M:%S')
file_handler.setFormatter(file_format)
logger.addHandler(file_handler)
```

This is what the outputs look like.

```python
2023-04-03 18:54:03 DEBUG: Debug message. - c:\users\agad069\desktop\logger\example2.py:46
2023-04-03 18:54:03 INFO: Info message. - c:\users\agad069\desktop\logger\example2.py:47
2023-04-03 18:54:03 WARNING: Warn message. - c:\users\agad069\desktop\logger\example2.py:48
2023-04-03 18:54:03 ERROR: Error message. - c:\users\agad069\desktop\logger\example2.py:49
2023-04-03 18:54:03 CRITICAL: Critical message. - c:\users\agad069\desktop\logger\example2.py:50
```

Consider clearing the handlers if necessary.

```python
logger.handlers.clear()
```

### Log to Both the Console and a File

This is an example to create a single Logger associated with 2 handlers:

1. A file handler.
2. A stream handler.

```python
import logging

level = logging.DEBUG
name = 'logfile.txt'

logger = logging.getLogger(name)
logger.setLevel(level)

file_handler = logging.FileHandler(name,'a+','utf-8')
file_handler.setLevel(logging.DEBUG)
file_format = logging.Formatter('%(asctime)s %(levelname)s: %(message)s - %(pathname)s:%(lineno)d', datefmt='%Y-%m-%d %H:%M:%S')
file_handler.setFormatter(file_format)
logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_format = logging.Formatter('%(message)s')
console_handler.setFormatter(console_format)
logger.addHandler(console_handler)
```

When a log message is executed, then it is both printed to the console and saved in the `logfile.txt`.

Consider clearing the handlers if necessary.

```python
logger.handlers.clear()
```

### PyGAD Example

To use the logger in PyGAD, just create your custom logger and pass it to the `logger` parameter.

```python
import logging
import pygad
import numpy

level = logging.DEBUG
name = 'logfile.txt'

logger = logging.getLogger(name)
logger.setLevel(level)

file_handler = logging.FileHandler(name,'a+','utf-8')
file_handler.setLevel(logging.DEBUG)
file_format = logging.Formatter('%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
file_handler.setFormatter(file_format)
logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_format = logging.Formatter('%(message)s')
console_handler.setFormatter(console_format)
logger.addHandler(console_handler)

equation_inputs = [4, -2, 8]
desired_output = 2671.1234

def fitness_func(ga_instance, solution, solution_idx):
    output = numpy.sum(solution * equation_inputs)
    fitness = 1.0 / (numpy.abs(output - desired_output) + 0.000001)
    return fitness

def on_generation(ga_instance):
    ga_instance.logger.info(f"Generation = {ga_instance.generations_completed}")
    ga_instance.logger.info(f"Fitness    = {ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]}")

ga_instance = pygad.GA(num_generations=10,
                       sol_per_pop=40,
                       num_parents_mating=2,
                       keep_parents=2,
                       num_genes=len(equation_inputs),
                       fitness_func=fitness_func,
                       on_generation=on_generation,
                       logger=logger)
ga_instance.run()

logger.handlers.clear()
```

By executing this code, the logged messages are printed to the console and also saved in the text file. 

```python
2023-04-03 19:04:27 INFO: Generation = 1
2023-04-03 19:04:27 INFO: Fitness    = 0.00038086960368076276
2023-04-03 19:04:27 INFO: Generation = 2
2023-04-03 19:04:27 INFO: Fitness    = 0.00038214871408010853
2023-04-03 19:04:27 INFO: Generation = 3
2023-04-03 19:04:27 INFO: Fitness    = 0.0003832795907974678
2023-04-03 19:04:27 INFO: Generation = 4
2023-04-03 19:04:27 INFO: Fitness    = 0.00038398612055017196
2023-04-03 19:04:27 INFO: Generation = 5
2023-04-03 19:04:27 INFO: Fitness    = 0.00038442348890867516
2023-04-03 19:04:27 INFO: Generation = 6
2023-04-03 19:04:27 INFO: Fitness    = 0.0003854406039137763
2023-04-03 19:04:27 INFO: Generation = 7
2023-04-03 19:04:27 INFO: Fitness    = 0.00038646083174063284
2023-04-03 19:04:27 INFO: Generation = 8
2023-04-03 19:04:27 INFO: Fitness    = 0.0003875169193024936
2023-04-03 19:04:27 INFO: Generation = 9
2023-04-03 19:04:27 INFO: Fitness    = 0.0003888816727311021
2023-04-03 19:04:27 INFO: Generation = 10
2023-04-03 19:04:27 INFO: Fitness    = 0.000389832593101348
```
