# Shepherd

Shepherd is a 3D visual object mapping system for robot querying.

## Documentation

- [Class Diagram](doc/class_diagram.puml)
- [Pipeline Diagram](doc/pipeline_diagram.puml)


## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

## Running

### Query Demo

This demo runs the 3D object query using an image.

```bash
python demo/query_demo.py
```

### Pipeline Demo

This demo runs each step of the pipeline in sequence so that you can see the intermediate results.

```bash
python demo/step_demo.py
```