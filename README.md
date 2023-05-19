# 矩阵区间压缩

## Quick start

```python
from range_compression import RangeCompressedMask, mask_encode
from pathlib import Path


mtx = .... # 带有很多连续值的矩阵
rcm = mask_encode(mtx)

X, Y = ..., ... # 要查找的 X, Y
res = rcm.find_index(X, Y)

assert res.shape == X.shape
assert (mtx[Y, X] == res).all()
```


## TODO

[ ] 把性能测试添加到测试和 readme 中，每个版本做性能回归测试  
[ ] 添加更多说明和直接能运行的快速入门
