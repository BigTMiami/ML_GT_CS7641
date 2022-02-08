Anthony Menninger
CS 7641 Machine Learning
Assignment 1

Infrastucture Setup
* Uses Python 3.8 (May work with 3.6 or 3.7) 
* CODE: https://github.com/BigTMiami/ML_GT_CS7641
    * To download, run: git clone https://github.com/BigTMiami/ML_GT_CS7641.git
    * The code for this is in the Assignment_1 folder
* DATA:Due to the size, the data was not loaded to Git, but can be pulled directly from the follwing urls
    * Census data: https://archive-beta.ics.uci.edu/ml/datasets/census+income+kdd
        * This should be loaded into Assignment_1/data/census
    * MNIST data: http://yann.lecun.com/exdb/mnist/
        * This should be loaded into Assignment_1/data/mnist
* Python Setup: In Assignment_1 is a requirements.txt.  This will load all the needed python libraries.
    * Best practice is to use a virtual environment, such as virtualenv, to load libraries.
    * From the command line in the Assignment_1 directory, install with command: pip install -r requirements.txt

Code base
* All of the code is in the Assignment_1 folder
* Data prep setup if data was not downloaded into Assignment_1/data/
    * MNIST_DATA_LOCATION must be set in mnist_data_prep.py file
    * CENSUS_DATA_LOCATION must be set in prep_census_data.py file
* The code produces all charts in the Assignment_1/document/figures/working directory
* The best place to start is the summary folder, which has the final review for all algorithms
    * The experiment results are presaved in the overal_results.py
    * overall_chart.py can be run to generate the charts using the overall_results.py
    * overall_review.py can be run to recreate the results.  
* Each algorithm type has it's own folder: decision_tree, boosting, etc.
    * decision_tree
        *decision_tree.py will create the charts for both datasets.
    * neural
        * census_neural_multiprocessing.py will produce the census analysis.  This uses the DQN_ files ansd was setup to use multiprocessing.
        * mnist_training.py will produce the MNIST analysis.  This uses the mnist_ files.
    * boosting
        * boosting_trainer.py will produce analysis for both data sets
    * svm 
        * svm_training.py will produce analysis for both data sets
    * knn 
        * knn_training.py will produce the analysis for both data sets

