# WebPredictor
This repository is intended to illustrate how to use Docker to run a simple web application using NiceGUI. This page allows you to upload a CSV file with a target column with missing data, and it will be automatically filled in.

## Table of Contents
- [How to run the web](#how-to-run-the-web)
   - [Using a conda environment](#using-a-conda-environment)

- [How to use the web](#how-to-use-the-web)

## How to run the web
You can do it using a conda environment or building a Docker image.

### Using a conda environment

1. Clone the repository:
```bash
git clone https://github.com/jcaste05/WebPredictor.git
```

2. Navigate to the project directory:
```bash
cd WebPredictor
```

3. Create a conda environment:
```bash
conda create -n webpredictor python=3.10
```

4. Activate the environment:
```bash
conda activate webpredictor
```

5. Install the requirements:
```bash
pip install -r requirements.txt
```

6. Add `PYTHONPATH` if it is necessary:

For windows:

```bash
$env:PYTHONPATH = $PWD.Path
```

For Linux/Ubuntu:

```bash
export PYTHONPATH=$PWD
```

7. Run the web application:
```bash
python ./web/main.py
```

### Using Docker

1. Clone the repository:
```bash
git clone https://github.com/jcaste05/WebPredictor.git
```

2.  Navigate to the project directory:
```bash
cd WebPredictor
```

3. Build the Docker image:
```bash
docker build -t webpredictor:latest .
```

4. Run the Docker container:
```bash
docker run -p 8080:8080 --name webpredictor webpredictor:latest
```

## How to use the web
1. Open your web browser and go to `http://localhost:8080`.

2. Upload your `CSV` with the column with missing data you want to predict.

3. Select the column which will be predicted.

4. Wait for a pop up indicating that certain number of rows has been predicted.

5. You can click the download button to get the original CSV with the predictions.
