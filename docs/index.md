# Mic Toolkit


![PyPI - Version](https://img.shields.io/pypi/v/mic-toolkit)
![Python](https://img.shields.io/badge/python-3.12-blue.svg)
![PyPI - Downloads](https://img.shields.io/pypi/dm/mic-toolkit)
[![CI](https://github.com/micvitc/mic-toolkit/actions/workflows/ci.yaml/badge.svg)](https://github.com/micvitc/mic-toolkit/actions/workflows/ci.yaml)



## Overview

Simple synthetic data generation using LLMs.

## Features

- **Text Generation**
- **Label Generation**

## Installation

To install Mic Toolkit, use the following command:

```sh
pip install mic-toolkit
```

## Current Utilities

### Simple Text Generation

``` py title="text-gen-sample.py"


import pandas as pd
from mic_toolkit.synthetic.generation import Generator


generator = Generator(endpoint="http://localhost:11434", model="llama3.2:3b-instruct-q4_0")

data = pd.read_csv("data.csv")

output = generator.generate_text(data, system_prompt="Translate the following text to French")

print(output)

```
Output

```
                                                text
0  He heard the crack echo in the late afternoon ...
1  There wasn't a bird in the sky, but that was n...
2  The choice was red, green, or blue. It didn't ...
3  What was beyond the bend in the stream was unk...
4  I guess we could discuss the implications of t...
5  There were about twenty people on the dam. Mos...

                                              output
0  Il entendit le bruit de craquement échoer à la...
1  Il n'y avait pas d'oiseau dans le ciel, mais c...
2  La choix était rouge, vert ou bleu. Il n'a pas...
3  Ce qui se trouvait au-delà de la courbe du rui...
4  Quelles implications auraient la phrase "c'est...
5  Il y avait environ vingt personnes sur la barr...

```

### Label Generation


``` py title="text-gen-sample.py"


import pandas as pd
from mic_toolkit.synthetic.generation import Generator


generator = Generator(endpoint="http://localhost:11434", model="llama3.2:3b-instruct-q4_0")

data = pd.read_csv("data.csv")

output = generator.generate_labels(data=data, labels=["Europe", "Asia", "Africa", "America", "Oceania"], query="Which continent does the following country belong to?")

print(output)

```

Output

```
           country    label
0           France   Europe
1        Argentina  America
2    United States  America
3           Canada  America
4           Mexico  America
5           Brazil  America
6   United Kingdom   Europe
7          Germany   Europe
8            Italy   Europe
9            Spain   Europe
10       Australia  Oceania
11           Japan     Asia

```