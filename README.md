# Optimization of Vehicle Platoons’ Cohesion and Environmental Impact in Urban Areas

This the code supporting the paper:

**Optimization of Vehicle Platoons’ Cohesion and Environmental Impact in Urban Areas**

Natalia Selini Hadjidimitriou, Alberto Compagnoni, Marco Mamei, Marco Picone and Ralf Willenbrock

Submitted to Scientific Reports in 2023

Files and data allow to reproduce all the reported experiments


## File content and structure

- **data** contains the main datasets to run the experiments. This fonder contains platoons' route recorded for the experiments

- **graph** contains jupyetr code to run experiments

	- *analysisXpaper.ipynb* produces Figure 3 and computed the main results (DIST and CO2 avearge savings)
	- *platoon-graph.ipynb* produces Figures 5,6,7,9
	- *violinplots.ipynb* produces Figure 1 and Figure 8 in the paper

- **regression** contains jupyetr code to run the regression of the paper. 
	- *platoon_v13_regr.ipynb* also contains the code to create Table 2 in the paper

- **results** the computed regression models are saved in this folder

- **routing** contains data and code to run routing experimtns. In particular:

	- *notebooks/src/platoon.py* contains useful functions and the weights of road features in the regression
	- *notebooks/platoon_routing.ipynb* contains the code to run simulations. This produces Figure 2 in the paper, and the file df.csv to be used by *analysisXpaper.ipynb*


