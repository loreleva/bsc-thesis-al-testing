# Introduction
This is the software testing for the active learning algorithm described in my Bachelor thesis.
The program command takes in input: 
- ```ntest```: the number of tests to do for each different percentage of the paths
- ```p_expansion```: the probability p of success of the binomial distribution on the size of the expansion of a label in a domain of a feature.

The other inputs are given through the JSON file ```input.json```.

# Usage
```
./test.sh ntest p_expansion
```

# Warnings
The test can be very slow if high values of the parameters are used
