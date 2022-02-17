Working in a command line environment is recommended for ease of use with git and dvc. If on Windows, WSL1 or 2 is recommended.

# Github Link
[link](https://github.com/r-judeh/nd0821-c3-starter-code)

# Environment Set up
* Download and install conda if you don’t have it already.
    * Use the supplied requirements file to create a new environment, or
    * conda create -n [envname] "python=3.8" scikit-learn dvc pandas numpy pytest jupyter jupyterlab fastapi uvicorn -c conda-forge
    * Install git either through conda (“conda install git”) or through your CLI, e.g. sudo apt-get git.

## Repositories
* Create a directory for the project and initialize git and dvc.
    * As you work on the code, continually commit changes. Generated models you want to keep must be committed to dvc.
* Connect your local git repo to GitHub.
* Setup GitHub Actions on your repo. You can use one of the pre-made GitHub Actions if at a minimum it runs pytest and flake8 on push and requires both to pass without error.
    * Make sure you set up the GitHub Action to have the same version of Python as you used in development.
* Set up a remote repository for dvc.

# Data
* Census data provided by UCI.
* This data was cleaned to remove white spaces.
* All the data is versioned and tracked using dvc and stored on an S3 bucket.

# Model
* A logistic regression model from sklearn is used to predict salary.
* A model card is provided for more information about the model.

# API Creation
*  A RESTful API using FastAPI was created and implements the following:
    * GET on the root giving a welcome message.
    * Post on /model_inference path that does inference

# API Deployment
* The API is deployed using Heroku
