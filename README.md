# Clustering assignment
Poznań University of Technology assignment for Data Mining class

## Dataset
https://www.kaggle.com/competitions/tabular-playground-series-jul-2022/overview

## Set up
Create virtual environment and install dependencies:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Using DVC
To fetch data from DVC remote storage:
```bash
dvc pull -r origin
```

## To add new data file
```bash
dvc add data/<file_name>
git commit -m "Add <file_name>"
dvc push -r origin
```

## To switch between versions
```bash
git checkout <commit/branch>
dvc checkout
```