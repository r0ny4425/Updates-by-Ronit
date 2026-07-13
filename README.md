# SimYuj

[![CI](https://github.com/riitk/simyuj/actions/workflows/ci.yml/badge.svg)](https://github.com/riitk/simyuj/actions/workflows/ci.yml)

SimYuj is a Python library for deterministic, event-driven quantum network
simulation. It provides the timeline engine, quantum-state services, network
components, resource bookkeeping, and reporting primitives used to assemble
protocol experiments.

The repository is organized as a reusable simulator library. Protocol runners,
experiment scripts, and scenario-specific control logic can live outside the
core package while sharing the same engine and component models.

## Project Status

SimYuj is in early 0.x development. The simulator is usable for experiments,
tutorials, and research prototypes, but public APIs and subsystem behavior may
change before a 1.0 release.

The current package version is `0.1.0`.

## Install

SimYuj requires Python 3.11 or newer.

```bash
git clone https://github.com/riitk/simyuj.git
cd simyuj
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\activate
```

For development and docs work:

```bash
pip install -e ".[dev,docs]"
```

## Check The Install

```bash
python -c "import simyuj; print(simyuj.__version__)"
pytest
```

## What Is In The Library

| Area | Purpose |
| --- | --- |
| `simyuj.engine` | Deterministic timeline, events, cancellation, execution summaries, and RNG streams. |
| `simyuj.qstate` | Quantum-state records, storage, math, noise, measurement, sampling, and validation. |
| `simyuj.components` | Ports, connections, channels, sources, detectors, memories, and quantum target routing. |
| `simyuj.control` | Event-driven agents, runtime context, classical actions, payloads, timers, reports, and device/resource services. |
| `simyuj.network` | Nodes, links, topology, routes, and path planning helpers. |
| `simyuj.resources` | Memory references, ownership, reservations, route requirements, and resource bookkeeping. |
| `simyuj.entanglement` | Entangled-pair records, registry state, construction helpers, and query utilities. |
| `simyuj.primitives` | Shared IDs, metadata, units, subsystem handles, validation helpers, and transport messages. |
| `simyuj.signal` | Signal records used to carry quantum/classical delivery metadata through components. |
| `simyuj.runtime` | Runtime binding helpers for preparing bindable simulator objects before execution. |
| `simyuj.metrics` | Link and route scoring helpers. |
| `simyuj.tracing` | Structured records, levels, loggers, and sinks for inspecting simulation runs. |

## Documentation

The Sphinx docs are organized by subsystem:

- [Getting started](docs/source/getting_started.rst)
- [Engine](docs/source/engine/index.rst)
- [Quantum state](docs/source/qstate/index.rst)
- [Components](docs/source/components/index.rst)
- [Network](docs/source/network/index.rst)
- [Resources](docs/source/resources/index.rst)
- [Metrics](docs/source/metrics/index.rst)
- [Tracing](docs/source/tracing/index.rst)

Build the docs locally with:

```bash
python -m sphinx -b html docs/source docs/build/html
```

## Development

Run the test suite:

```bash
pytest
```

Run the usual local checks:

```bash
black .
isort .
flake8 .
mypy
```

Install pre-commit hooks if you want the checks to run before each commit:

```bash
pre-commit install
```

## Repository Layout

```text
simyuj/
├── src/simyuj/        # installable library
├── tests/             # regression tests
├── docs/              # Sphinx documentation
├── examples/          # small usage examples
├── CHANGELOG.md
├── CITATION.cff
├── pyproject.toml
├── README.md
├── CONTRIBUTING.md
└── LICENSE
```

## Contributing

Start with [CONTRIBUTING.md](CONTRIBUTING.md). It covers the development
workflow and project conventions.

## License

SimYuj is distributed under the MIT License. See [LICENSE](LICENSE).
