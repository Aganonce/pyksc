# Python K-SC Implementation

This program is a rudimentary python implementation of K-SC built for maximum customizability.[1]

# Running
First run `pip install -r requirements.txt`.

To run K-SC, follow the example below:
```python
from KSC import KSC

res, cent = KSC(k=3, iname="data/MemePhr.txt", delta=5, oname="results/MemePhr", plot=True)
```

## References
[1]: Yang, Jaewon, and Jure Leskovec. "Patterns of temporal variation in online media." Proceedings of the fourth ACM international conference on Web search and data mining. ACM, 2011.
