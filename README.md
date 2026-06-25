# AMD Hackathon Agent

A structured, production-oriented agent project built for the AMD Hackathon. This repository contains the code, tooling, and workflows for running and evaluating an AI agent system.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Environment Variables](#environment-variables)
- [Usage](#usage)
- [Development](#development)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

This project is an AI agent implementation created for hackathon workflows. It is primarily Python-based, with smaller Rust/C components for performance-focused or systems-level tasks.

Use this repository to:

- Run the agent locally
- Iterate on prompts, tools, and logic
- Test behavior and performance
- Prepare hackathon-ready demos and submissions

## Features

- Modular agent workflow design
- Python-first development experience
- Extensible architecture for tools/integrations
- Mixed-language support for performance-critical pieces
- Ready for iterative experimentation

## Tech Stack

Language composition (from repository metadata):

- **Python**: 94.7%
- **Rust**: 5.0%
- **C**: 0.2%
- **Cython**: 0.1%
- **Go / Makefile**: minimal

## Repository Structure

> Update this section as files evolve.

```text
.
├── src/                # Core agent source code
├── scripts/            # Utility / automation scripts
├── tests/              # Test suite
├── rust/               # Rust components (if applicable)
├── docs/               # Project docs and design notes
└── README.md           # Project overview and setup guide
```

## Getting Started

### Prerequisites

- Python 3.11+ (recommended)
- `pip` or `uv` for dependency management
- (Optional) Rust toolchain for Rust modules

### Installation

```bash
# 1) Clone repository
git clone https://github.com/Hour-Meng/amd-hackathon-agent.git
cd amd-hackathon-agent

# 2) Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\\Scripts\\activate    # Windows PowerShell

# 3) Install dependencies
pip install -U pip
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the repository root (if your workflow requires API keys):

```bash
cp .env.example .env
```

Common variables (example):

```env
OPENAI_API_KEY=your_key_here
MISTRAL_API_KEY=your_key_here
```

## Usage

Run the main entry point (adjust to your project’s actual entry file):

```bash
python -m src.main
```

Or, if you use a script-based entrypoint:

```bash
python run.py
```

## Development

### Run tests

```bash
pytest -q
```

### Format and lint (example)

```bash
ruff check .
ruff format .
```

## Troubleshooting

- **ModuleNotFoundError**: Ensure your virtual environment is activated and dependencies are installed.
- **Key/auth errors**: Verify `.env` values and exported environment variables.
- **Native build issues**: Reinstall build tools and verify Rust/C toolchains are available.

## Contributing

Contributions are welcome.

1. Fork the repo
2. Create a feature branch
3. Commit your changes
4. Open a pull request

## License

Add your license information here (e.g., MIT, Apache-2.0).
