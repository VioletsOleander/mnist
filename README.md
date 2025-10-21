# MNIST

A toy based on libtorch API.

## 1. Prerequisite

This project is managed by [Pixi](https://github.com/prefix-dev/pixi/). Before moving on, please ensure it is installed already.

## 2. Getting Started

### 2.1 Get Data

Get dataset and do preprocess:

```bash
pixi install -e dataset
pixi r get-dataset
pixi r process-dataset
```

### 2.2 Configure and Build

Configure and build the project:

```bash
pixi install -e default
pixi r configure [release | debug]
pixi r build
```

The task `configure` defaults to `debug`. For release configure and build, type:

```bash
pixi r configure release
```

### 2.3 Train and Test

Train the network:

```bash
pixi r train <path-to-train-config-file>
```

For example:

```bash
pixi r train configs/train.toml
```

The `<path-to-train-config-file>` defaults to `configs/train.toml`.

Test the trained network:

```bash
pixi r test <path-to-test-config-file>
```

For example:

```bash
pixi r test configs/test.toml
```

The `<path-to-test-config-file>` defaults to `configs/test.toml`.

Example config file can be found in `configs/config.example.toml`.
