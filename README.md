# -Integrative Think Tank 20 (ITT 20)
For more information on how ITTs work see https://samba.ac.uk/working-with-samba/integrative-think-tanks-itts/.
# Reverse engineering atompsheric dust from aircraft engine samples (Finding airport dust composition)
This folder consists of group work done during ITT 20 from June 10th-14th 2024. The challenge was determining mathematical methods to predict airport dust composite concentration based off 20 decomissioned aircraft engines and their final dust composite concentrations. For more detail see the ITT 20 reports.

## Functionality of the files and folders
 Folders:
  - data - holds the 20 engine dataset with their respective flights taken (3 CSV files)
  - linear_systems - contains two Python files using the dataset 
  - bayesian - contains Python files (jupyter notebooks) on a toy problem with 2-4 airports and finding 2 dust type concentrations using MCMC

Main files:
- overlap.py - solving the linear_systems approach with the least-squares and constrained optimisation approaches
- limit_data.py - an extension of overlap.py with the full methods and their descriptions with the plots (heatmap of the absolute error between the dust types predictions and ground truth and the respective error pdf plots)
- flight_toy_one, flight_toy_two_one, flight_toy_two_two Jupyter notebooks - Bayesian approach in action on the toy problems


## Libraries
In this project, several Python libraries were used for model construction, mathematical calculations, plotting and data manipulation (including the standard numpy, matplotlib, pandas, scipy etc) 

## Interface instructions
To create the conda environment one can install it via a dependencies file command:
```
conda env create -f dependencies.yml
```
This exports the relevant libraries into a new environment called my_env_project, which can then be activated by:
```
conda activate my_env_project
```

Then, the code can be run through the command line for example
```
python linear_systems/limit_data.py
```

or using coding editors such as VSC to run the individual files. 

# Acknowledgements
This was part of a group working at ITT 20 with Amin Sabir, Bill Nunn, Daniel Hajnal and Matt Evans. In addition, Dr Theresa Smith, Dr Sergey Dolgov and Prof Matt Nunes helped out massively with insightful discussions about methods to implement and how they would work. The ITT 20 partner Rolls-Royce Holdings was very helpful in proposing this challenge in the first place, dataset for use and helping to answer any queries.

This project was part of the ITT20 and the 1st year (MRes) for the statistical applied mathematics at the University of Bath (SAMBa) PhD programme.































