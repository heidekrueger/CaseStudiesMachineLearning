## Code Folder

This directory contains mostly scripts for testing the algorithms on dataset and performing analysis on the outputs. 

The actual implementations of the algorithms can be found in the subdirectories
`code/ProximalMethod/ ` and `code/SQN/`

### The following is outdated.

Aufbau:

	SQN
	LogisticRegression
	stochastic_tools

Testen:
	python tests.py [#test]
	
	1: SQN deterministic
	2: SQN stochastic using Logistic Regression and benchmarking
	3: SQN on Logistic Regression