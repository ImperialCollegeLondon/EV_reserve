## Reserve Provision from Electric Vehicles: Aggregate Boundaries and Stochastic Model Predictive Control

The code in this repository simulates a Stochastic Model Predictive Control (SMPC) algorithm for an aggregate electric vehicle (EV) fleet. The algorithm was proposed as part of a journal submission for which a [preprint](https://doi.org/10.48550/arXiv.2406.07454) has been made available.

# Input data

The required input data has been uploaded alongside the results as a [dataset](https://dx.doi.org/10.21227/ckmr-1g77) on IEEE dataport but the individual sources are listed below:
1. Plug-in times and energy charged: Department for Transport, “Electric chargepoint analysis 2017: Domestics,” <https://www.data.gov.uk/dataset/5438d88d-695b-4381-a5f2-6ea03bf3dcf0/electric-chargepoint-analysis-2017-domestics>, 2018, accessed: 2023-09-15.
2. Rain volume: L. V. Alexander and P. D. Jones, “Updated precipitation series for the UK and discussion of recent extremes,” Atmospheric science letters, vol. 1, no. 2, pp. 142–150, 2000.
3. D. E. Parker, T. P. Legg, and C. K. Folland, “A new daily central england temperature series, 1772–1991,” International journal of climatology, vol. 12, no. 4, pp. 317–342, 1992.
4. Elexon, “Balancing mechanism reporting service (bmrs): Market index data,” <https://www.bmreports.com/bmrs/?q=balancing/marketindex/historic>, 2023, accessed: 2023-09-15.

Datasets 2-4 have also been made available in this repository but the electric chargepoint analysis (datast 1) is too large to be uploaded on GitHub.

# Functionality

`a_remove_double_occ.py` removes plug-in entries in which a charger is occupied by two EVs at the same time as this is not possible. It also removes any charging process that lasts longer than two weeks. Note that this process is computationally intensive, so that it should only be done once.
`b_add_EVdata.py` formats the plug-in data and adds some additional measures as described in the paper.
`c_random_seed.py` uses a charger list that was created using a one-time random seed to create datasets of 20-1000 EV chargers.
`d_aggregate_bounds.py` creates a function that aggregates the data from any of the charging datasets into the three continuous boundaries (Power Boundary, Upper Energy Boundary, Lower Energy Boundary)
`e_predmodel.py` contains a multiple linear regression (MLR) model that uses the boundary trajectories from the training dataset to predict the boundary trajectories in the test dataset. Additionally, the boundary scenarios are also generated here.
`f_stoch_opt.py` presents the first- and second-stage optimisation functions which take the generated predictions and scenarios as an input and generate charging decisions. The first-stage optimisation additionally decides on reserve provision.
`g_smpc.py` iterates through half-hour settlements, calling the previous file at each timestep to solve the resulting optimisation.
`simulations.ipynb` runs simulations for different fleet sizes. It is recommended to use some sort of parallel processing here, such as the [multiprocessing package in Python](https://docs.python.org/3/library/multiprocessing.html#module-multiprocessing) as otherwise the simulations will be very time-consuming.
`auxfunc_aggregation.py`, `auxfunc_data.py` and `auxfunc_test_train_split.py` are files with auxiliary functions that are called upon for the aggregation process, data processing and the test/train split, respectively.