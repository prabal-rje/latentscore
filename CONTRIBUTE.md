## Setting Up Your Environment with Conda

### 1. Install Conda

If you don't have Conda installed already, download and install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution) following the instructions for your operating system.

### 2. Create or Update the Environment

Create the `latentscore` environment from the repo's `environment.yml` (arm64 Conda on Apple Silicon):

```bash
conda env create -f environment.yml
```

If the environment already exists, update it:

```bash
conda env update -f environment.yml --prune
```

### 3. Activate the Environment

Before installing packages, activate the new environment:

```bash
conda activate latentscore
```

### 4. Editable Install (Optional)

If you want editable imports for development:

```bash
pip install -e .
```

### 5. Next Time You Want To Contribute

You only need to **create and set up** the environment once! The next time you want to contribute, just activate the environment:

```bash
conda activate latentscore
```

Now you are ready to start working!

## Development Loop

Install project dependencies and tooling via `environment.yml` (see above). Before opening a pull request, run the full suite:

```bash
make check
```
