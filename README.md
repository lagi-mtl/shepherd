# Shepherd

Shepherd is a computer vision pipeline and world modeling system designed for easy integration in embodied systems. Although initially built for Rove, [Capra](https://www.clubcapra.com)'s autonomous search and rescue robot, it aims to be flexible and lightweight enough to be run on any platform. It detects, tracks, and analyzes objects to gradually build a semantic 3D representation of its environment by maintaining a persistent database of object point clouds with associated visual and semantic features, enabling spatial reasoning and natural language queries about the observed scene. It is heavily inspired by [ConceptFusion](https://concept-fusion.github.io) and developped with equal contribution by Simon Roy and Samuel Barbeau.

***It is currently a work in progress.***

## TODO

- [ ] Add docker support for easier deployment
- [ ] Improve inference speed
- [ ] Add live 3D visualization
- [ ] Benchmark inference speed and memory usage on jetson

## Documentation

- [Class Diagram](doc/class_diagram.puml)
- [Pipeline Diagram](doc/pipeline_diagram.puml)


## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

## Running

As mentionned above, Sherpherd is currently in development and is likely not bug free. We therefore cannot guarantee that the demos will work on all systems.

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

## References

This work builds upon:
- [ConceptGraph: A Graph Neural Network Framework for Building Robot World Models](https://concept-graphs.github.io/)
