# Classification and clustering pipeline for neuronal morphologies

## Installation

Add as a Git Submodule:

```bash
git submodule add git@github.com:laurapd/classification_pipeline.git external/classification_pipeline
cd external/classification_pipeline
pip install -e .
```

Then, import the code accordingly in your scripts:
```py
from ccp import classify_run_net, cluster_run_net
```