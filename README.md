# Shepherd

Shepherd is a computer vision pipeline and world modeling system designed for easy integration in embodied systems. It detects, tracks, and analyzes objects to gradually build a semantic 3D representation of its environment by maintaining a persistent database of object point clouds with associated visual and semantic features, enabling spatial reasoning and natural language queries about the observed scene. It is heavily inspired by [ConceptGraph](https://concept-graphs.github.io/) and developped with equal contribution by Simon Roy and Samuel Barbeau.

***It is currently a work in progress.***

## TODO

- [ ] Fix world coordinate transformations for consistent object positioning
- [ ] Optimize point cloud merging by utilizing the database querying
- [ ] Implement LLM for reverse-querying
- [ ] Add reverse-querying based on object captions
- [ ] Add path finding based on reverse-query

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

## References

This work builds upon:
- [ConceptGraph: A Graph Neural Network Framework for Building Robot World Models](https://concept-graphs.github.io/)

This demo runs the Habitat simulator with the Shepherd pipeline. To run this demo, you first need to download the Replica data and place it at the root of the repository. (See the [Meta Repository](https://github.com/facebookresearch/Replica-Dataset))

```bash
python demo/habitat_demo.py
```
