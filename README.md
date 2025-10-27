# ai-traffic-signal

A small collection of algorithms used during thesis experiments: RRT*, Dijkstra's grid shortest path, and a weighted round-robin helper.

Install (development):

```bash
python -m pip install -e .[dev]
```

Basic usage:

```python
from ai_traffic_signal import load_module
rrt = load_module('rrt_dijkstra')
```

See the modules for function docs.
