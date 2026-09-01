# BrainCapture Product Template Repository

[![security: bandit](https://img.shields.io/badge/security-bandit-yellow.svg)](https://github.com/PyCQA/bandit)
[![Formatted with Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Tests](https://github.com/BrainCapture/product-template/actions/workflows/test.yml/badge.svg)](https://github.com/BrainCapture/product-template/actions/workflows/test.yml)
[![Python versions](https://img.shields.io/badge/python-3.12-blue)](https://www.python.org/downloads/release/python-3120/)

This repository serves as a golden image for product projects at BrainCapture. It provides a standardized structure and essential components to kickstart new projects efficiently.

## Prerequisites

1. A Google service account JSON key <google-service-account-key.json> placed in your project Remember to set the exact path in the `.env` file, see `.env.example` for reference.
2. [Taskfile](https://taskfile.dev/) v3 installed (used to run predefined tasks). This will be installed when you run the `/scripts/setup.sh` script.
3. [uv](https://docs.astral.sh/uv/getting-started/installation/) should be installed (used to manage python packages)

## Setup
To set up the project, run the following command in your terminal:

```bash
bash /scripts/setup.sh
```

This script will install all necessary dependencies and configure your environment.

## Usage
After setup, you can start using the project by running:

```bash
python main.py
```

or you can build the project Docker image using:

```bash
task build
```

and run it with:

```bash
task run
```

## Contributing
Contributions are welcome! Please create a pull request or open an issue to discuss changes.
If you have any questions, feel free to reach out to Magnus at mgp@braincapture.dk.

## License
This repository is for internal use at BrainCapture and is not publicly licensed.

2025 © BrainCapture AI. All rights reserved