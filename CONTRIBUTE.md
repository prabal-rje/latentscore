## Setting Up Your Environment with Conda

### 1. Install Conda

If you don't have Conda installed already, download and install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution) following the instructions for your operating system.

### 2. Create a New Environment

Open your terminal and create a new environment named `latentscore-3` (arm64 Conda on Apple Silicon):

```bash
conda create -n latentscore-3 python=3.10
```

*Feel free to set a different Python version if required.*

### 3. Activate the Environment

Before installing packages, activate the new environment:

```bash
conda activate latentscore-3
```

### 4. Install Requirements

If there's a `requirements.txt` file in this repository, install all dependencies with:

```bash
pip install -r requirements.txt
```

### 5. Next Time You Want To Contribute

You only need to **create and set up** the environment once! The next time you want to contribute, just activate the environment:

```bash
conda activate latentscore-3
```

Now you are ready to start working!

## Development Loop

Install project dependencies and tooling with:

```bash
pip install -r requirements.txt
```

Before opening a pull request, run the full suite:

```bash
make check
```
