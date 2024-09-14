# Welcome to the mic-toolkit

Utilities for internal MIC projects

## Installation

Make sure you have a python version>=3.12

``` bash

pip install mic-toolkit

```

## Current Utilities

### Simple Data Generation Pipeline

``` py title="sample.py"

from mic_toolkit.synthetic.generation import TextGenerationPipeline
from datasets import Dataset

dataset = Dataset.from_dict(
    {"instruction": ["Write a Python program to multiply two numbers."]}
)

pipeline = TextGenerationPipeline(
    model_name="model_name",
    api_key="api_key",
    base_url="base_url",
)


distiset = pipeline.run_pipeline(dataset=dataset)

print(distiset["default"]["train"][0]["generation"])

```

