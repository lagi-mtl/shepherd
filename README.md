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

### Habitat Demo

This demo runs the Habitat simulator with the Shepherd pipeline. To run this demo, you first need to download the Replica data and place it at the root of the repository. (See the [Meta Repository](https://github.com/facebookresearch/Replica-Dataset))

```bash
python demo/habitat_demo.py
```