# Flower (Flwr) Federated Learning - Lessons Learned


**This document is based on real debugging experience. Save yourself hours of debugging by following these patterns!**

A comprehensive guide based on hands-on experience implementing federated learning with Flower 1.26.0.

---

## Table of Contents

1. [Version and Import Patterns](#1-version-and-import-patterns)
2. [API Parameter Naming Conventions](#2-api-parameter-naming-conventions)
3. [FedAvg Strategy Configuration](#3-fedavg-strategy-configuration)
4. [Client App Patterns](#4-client-app-patterns)
5. [Server App Patterns](#5-server-app-patterns)
6. [MetricRecord Key Names](#6-metricrecord-key-names)
7. [ArrayRecord Usage](#7-arrayrecord-usage)
8. [Configuration in pyproject.toml](#8-configuration-in-pyprojecttoml)
9. [Ray Backend Issues](#9-ray-backend-issues)
10. [Common Errors and Solutions](#10-common-errors-and-solutions)

---

## 1. Version and Import Patterns

### Correct Imports (Flower 1.26.0+)

```python
# Client App
from flwr.clientapp import ClientApp
from flwr.app import Context, Message, ArrayRecord, MetricRecord, RecordDict

# Server App
from flwr.serverapp import ServerApp, Grid
from flwr.serverapp.strategy import FedAvg
from flwr.app import Context, ArrayRecord, MetricRecord, ConfigRecord
```

### DEPRECATED Imports (DO NOT USE)

```python
# These are DEPRECATED - will show warnings
from flwr.client import ClientApp  # Wrong!
from flwr.server import ServerApp  # Wrong for new API!
from flwr.common import Context, Message  # Deprecated!
```

### Key Insight

> Importing from `flwr.server`, `flwr.client`, or `flwr.common` is deprecated.
> Use `flwr.serverapp`, `flwr.clientapp`, or `flwr.app` instead.

---

## 2. API Parameter Naming Conventions

### Use Hyphens, NOT Underscores

Flower uses **kebab-case** (hyphens) for configuration keys, not snake_case.

| Correct | Incorrect |
|---------|-----------|
| `num-server-rounds` | `num_server_rounds` |
| `learning-rate` | `learning_rate` |
| `batch-size` | `batch_size` |
| `local-epochs` | `local_epochs` |
| `num-partitions` | `num_partitions` |
| `num-examples` | `num_examples` |
| `fraction-train` | `fraction_train` |
| `partition-id` | `partition_id` |

### Where This Applies

1. **pyproject.toml** config keys
2. **MetricRecord** dictionary keys
3. **ConfigRecord** dictionary keys
4. **context.run_config** access
5. **context.node_config** access

---

## 3. FedAvg Strategy Configuration

### Correct Parameter Names (1.26.0+)

```python
from flwr.serverapp.strategy import FedAvg

strategy = FedAvg(
    fraction_train=1.0,        # NOT fraction_fit
    fraction_evaluate=0.5,     # This one is correct
    min_train_nodes=2,         # NOT min_fit_clients
    min_evaluate_nodes=2,      # NOT min_evaluate_clients
    min_available_nodes=2,     # NOT min_available_clients
)
```

### WRONG Parameter Names (Will Cause Errors)

```python
# These will cause TypeError!
strategy = FedAvg(
    fraction_fit=1.0,          # Wrong! Use fraction_train
    min_fit_clients=2,         # Wrong! Use min_train_nodes
    min_evaluate_clients=2,    # Wrong! Use min_evaluate_nodes
    min_available_clients=2,   # Wrong! Use min_available_nodes
)
```

### Error You'll See

```
TypeError: FedAvg.__init__() got an unexpected keyword argument 'fraction_fit'.
Did you mean 'fraction_train'?
```

---

## 4. Client App Patterns

### Complete Working Example

```python
from flwr.clientapp import ClientApp
from flwr.app import Context, Message, ArrayRecord, MetricRecord, RecordDict

app = ClientApp()

@app.train()
def train_fn(msg: Message, context: Context) -> Message:
    # Access configuration
    partition_id = context.node_config["partition-id"]  # Hyphen!
    batch_size = context.run_config["batch-size"]       # Hyphen!

    # Get config from server message
    learning_rate = msg.content["config"]["learning-rate"]  # Hyphen!

    # Load model from server arrays
    model = get_model()
    state_dict = msg.content["arrays"].to_torch_state_dict()
    model.load_state_dict(state_dict)

    # Train locally
    train_loss, num_examples = train(model, train_loader, ...)

    # Return updated model and metrics
    updated_arrays = ArrayRecord(model.state_dict())
    metrics = MetricRecord({
        "train_loss": float(train_loss),
        "num-examples": num_examples,  # MUST use hyphen!
    })

    content = RecordDict({
        "arrays": updated_arrays,
        "metrics": metrics,
    })

    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate_fn(msg: Message, context: Context) -> Message:
    # Similar pattern for evaluation
    ...
    metrics = MetricRecord({
        "eval_loss": float(loss),
        "eval_accuracy": float(accuracy),
        "num-examples": num_examples,  # MUST use hyphen!
    })

    content = RecordDict({"metrics": metrics})
    return Message(content=content, reply_to=msg)
```

---

## 5. Server App Patterns

### Complete Working Example

```python
from flwr.serverapp import ServerApp, Grid
from flwr.serverapp.strategy import FedAvg
from flwr.app import Context, ArrayRecord, MetricRecord, ConfigRecord

app = ServerApp()

@app.main()
def main(grid: Grid, context: Context) -> None:
    # Load configuration
    num_rounds = context.run_config["num-server-rounds"]
    fraction_train = context.run_config["fraction-train"]
    learning_rate = context.run_config["learning-rate"]

    # Initialize global model
    global_model = get_model()
    initial_arrays = ArrayRecord(global_model.state_dict())

    # Define server-side evaluation callback
    def server_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        model = get_model()
        state_dict = arrays.to_torch_state_dict()
        model.load_state_dict(state_dict)

        loss, accuracy, _ = evaluate(model, test_loader, device)

        return MetricRecord({
            "centralized_loss": float(loss),
            "centralized_accuracy": float(accuracy),
        })

    # Configure strategy
    strategy = FedAvg(
        fraction_train=fraction_train,
        fraction_evaluate=0.5,
        min_train_nodes=2,
        min_evaluate_nodes=2,
        min_available_nodes=2,
    )

    # Training config to send to clients
    train_config = ConfigRecord({
        "learning-rate": learning_rate,  # Hyphen!
    })

    # Start federated learning
    result = strategy.start(
        grid=grid,
        initial_arrays=initial_arrays,
        train_config=train_config,
        num_rounds=num_rounds,
        evaluate_fn=server_evaluate,
    )

    # Access final model
    final_state_dict = result.arrays.to_torch_state_dict()
```

---

## 6. MetricRecord Key Names

### Critical: Use `num-examples` with Hyphen

The FedAvg strategy uses `num-examples` (with hyphen) for weighted averaging.
If you use `num_examples` (underscore), you'll get this error:

```
flwr.serverapp.exception.InconsistentMessageReplies:
Missing required key `num-examples` in the MetricRecord of reply messages.
Cannot average ArrayRecords and MetricRecords. Skipping aggregation.
```

### Correct Usage

```python
metrics = MetricRecord({
    "train_loss": float(train_loss),
    "num-examples": num_examples,  # HYPHEN, not underscore!
})
```

### Why This Matters

The FedAvg strategy's `weighted_by_key` parameter defaults to `"num-examples"`.
It looks for this exact key to weight the aggregation.

---

## 7. ArrayRecord Usage

### Converting PyTorch State Dict

```python
from flwr.app import ArrayRecord
import torch

# Create from state_dict
model = get_model()
arrays = ArrayRecord(model.state_dict())

# Or explicitly
arrays = ArrayRecord.from_torch_state_dict(model.state_dict())

# Convert back
state_dict = arrays.to_torch_state_dict()
model.load_state_dict(state_dict)
```

### Accessing Arrays in Messages

```python
# In client, receiving from server
state_dict = msg.content["arrays"].to_torch_state_dict()

# In server, from strategy result
final_state_dict = result.arrays.to_torch_state_dict()
```

---

## 8. Configuration in pyproject.toml

### Complete Example

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "my-fl-project"
version = "1.0.0"
requires-python = ">=3.10"
dependencies = [
    "flwr[simulation]>=1.26.0",
    "flwr-datasets[vision]>=0.5.0",
    "torch>=2.2.0",
    "torchvision>=0.17.0",
]

[tool.hatch.build.targets.wheel]
packages = ["my_package"]

[tool.flwr.app]
publisher = "my-project"

[tool.flwr.app.components]
serverapp = "my_package.server_app:app"
clientapp = "my_package.client_app:app"

[tool.flwr.app.config]
# All keys use hyphens!
num-server-rounds = 10
fraction-train = 1.0
fraction-evaluate = 0.5
local-epochs = 2
batch-size = 32
learning-rate = 0.01
num-partitions = 10
alpha = 0.5
```

### Key Points

1. Use `flwr[simulation]` for local simulation with Ray
2. Use `flwr-datasets[vision]` for image dataset partitioning
3. All config keys use **hyphens**
4. Point to `module.file:app` for serverapp/clientapp

---

## 9. Ray Backend Issues

### Problem: Paths with Spaces

Ray workers fail when the project path contains spaces:

```
bash: /Users/name/Documents/AI: No such file or directory
bash: line 0: exec: /Users/name/Documents/AI: cannot execute
worker_pool.cc: Some workers have not registered within the timeout
```

### Solution 1: Copy to Path Without Spaces

```bash
cp -r "/path/with spaces/project" /tmp/fedvision_project
cd /tmp/fedvision_project
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
flwr run .
```

### Solution 2: Symlink (May Not Always Work)

```bash
ln -s "/path/with spaces/project" /tmp/fedvision
cd /tmp/fedvision
# Note: venv must also be in path without spaces
```

### Best Practice

**Always use paths without spaces for Flower projects.**

---

## 10. Common Errors and Solutions

### Error 1: Wrong FedAvg Parameters

```
TypeError: FedAvg.__init__() got an unexpected keyword argument 'fraction_fit'
```

**Solution:** Use `fraction_train`, `min_train_nodes`, etc.

---

### Error 2: Missing num-examples Key

```
InconsistentMessageReplies: Missing required key `num-examples` in the MetricRecord
```

**Solution:** Use `"num-examples"` (hyphen) not `"num_examples"` (underscore)

---

### Error 3: Import Deprecation Warnings

```
Importing from flwr.common is deprecated. Use flwr.app instead.
```

**Solution:** Update imports to use `flwr.clientapp`, `flwr.serverapp`, `flwr.app`

---

### Error 4: Ray Worker Crashes

```
worker_pool.cc: Some workers have not registered within the timeout
```

**Solution:** Check for spaces in path. Copy project to `/tmp/` or another path without spaces.

---

### Error 5: Message Content Access

```
KeyError: 'arrays'
```

**Solution:** Access through `msg.content["arrays"]`, not `msg.arrays`

---

## Quick Reference Card

```python
# Imports
from flwr.clientapp import ClientApp
from flwr.serverapp import ServerApp, Grid
from flwr.serverapp.strategy import FedAvg
from flwr.app import Context, Message, ArrayRecord, MetricRecord, RecordDict, ConfigRecord

# ClientApp
app = ClientApp()

@app.train()
def train_fn(msg: Message, context: Context) -> Message:
    state_dict = msg.content["arrays"].to_torch_state_dict()
    lr = msg.content["config"]["learning-rate"]
    partition_id = context.node_config["partition-id"]

    # ... train ...

    return Message(
        content=RecordDict({
            "arrays": ArrayRecord(model.state_dict()),
            "metrics": MetricRecord({"num-examples": n}),  # HYPHEN!
        }),
        reply_to=msg
    )

# ServerApp
app = ServerApp()

@app.main()
def main(grid: Grid, context: Context) -> None:
    strategy = FedAvg(
        fraction_train=1.0,      # NOT fraction_fit
        min_train_nodes=2,       # NOT min_fit_clients
    )
    result = strategy.start(
        grid=grid,
        initial_arrays=ArrayRecord(model.state_dict()),
        train_config=ConfigRecord({"learning-rate": 0.01}),
        num_rounds=10,
    )
```



