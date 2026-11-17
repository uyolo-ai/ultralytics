# Architecture Overview

Ultralytics is a Python computer-vision framework that exposes YOLO and related models through a shared Python API and `yolo` CLI. Its core pipeline connects task-specific models and data handling to reusable train, validation, prediction, tracking, and export workflows.

## Component diagram

```mermaid
flowchart TD
    User[User / application]
    CLI[CLI: ultralytics.cfg.entrypoint]
    API[Python API: ultralytics.YOLO and task model classes]
    Config[Configuration: ultralytics.cfg]
    Model[Shared model API: ultralytics.engine.model.Model]
    TaskModels[Task-specific models and task maps\nultralytics.models]
    NN[Network construction and modules\nultralytics.nn.tasks + nn.modules]

    User --> CLI
    User --> API
    CLI --> Config
    Config -->|parse task, mode, model and overrides| Model
    API --> Model
    Model --> TaskModels
    TaskModels --> NN

    subgraph Workflows[Model workflows]
        Train[Training\nultralytics.engine.trainer]
        Validate[Validation\nultralytics.engine.validator]
        Predict[Prediction and tracking\nultralytics.engine.predictor + trackers]
        Export[Model export\nultralytics.engine.exporter]
    end

    Model --> Train
    Model --> Validate
    Model --> Predict
    Model --> Export

    Data[Datasets, transforms and loaders\nultralytics.data]
    Train --> Data
    Validate --> Data
    Predict --> Data
    Data -->|batches / input tensors| Train
    Data -->|batches / input tensors| Validate
    Data -->|images, video, streams or tensors| Predict

    Backend[Inference backend selection\nultralytics.nn.autobackend.AutoBackend]
    Runtime[PyTorch and exported runtimes\nONNX, OpenVINO, TensorRT, CoreML, TFLite, ...]
    Predict --> Backend
    Backend --> Runtime
    Runtime -->|raw predictions| Predict
    Predict --> Results[Structured outputs\nultralytics.engine.results.Results]

    Loss[Task losses and metrics\nultralytics.utils]
    Train --> Loss
    Validate --> Loss
    Loss --> Train
    Loss --> Validate

    Checkpoints[Weights and run artifacts]
    Train -->|best.pt, last.pt, metrics and plots| Checkpoints
    Export -->|deployment model + metadata| Artifacts[Exported model artifacts]
```

## Runtime flow

- **Choose a task and mode:** The `yolo` entry point (`ultralytics.cfg.entrypoint`) parses task, mode, model and configuration overrides; the Python API offers the same workflow through model methods such as `train()`, `val()`, `predict()` and `export()`.
- **Dispatch to task implementations:** `ultralytics.engine.model.Model` uses its task map to select the relevant model, trainer, validator or predictor. `ultralytics.nn.tasks` builds neural networks from model YAML definitions and provides shared model/task behavior.
- **Load inputs and run work:** `ultralytics.data` supplies datasets, transforms, dataloaders and inference-source loaders. Training and validation consume dataset batches; prediction uses `BasePredictor` to preprocess inputs, run inference and package outputs as `Results` objects.
- **Deploy or continue training:** Prediction loads PyTorch or exported models through `AutoBackend`. Training writes checkpoints and metrics; `Exporter` converts PyTorch models to supported deployment formats and adds model metadata.

## Key package areas

| Path | Responsibility |
| --- | --- |
| `ultralytics/cfg/` | CLI entry point, task/mode selection, defaults and argument validation |
| `ultralytics/models/` | YOLO and other model families, including task-specific implementations |
| `ultralytics/engine/` | Shared model API, training, validation, prediction, results and export workflows |
| `ultralytics/nn/` | Model construction, neural-network modules and multi-format inference backends |
| `ultralytics/data/` | Dataset definitions, augmentation, loaders and dataloader construction |
| `ultralytics/utils/` | Shared configuration, checks, losses, metrics, callbacks and runtime utilities |
| `ultralytics/trackers/`, `ultralytics/solutions/` | Object tracking and higher-level computer-vision workflows |

## Extension points

Task model classes provide task maps that connect a task to its model, trainer, validator and predictor. Custom training, validation, prediction and export behavior can also be integrated through the callback hooks shared by these workflows.
