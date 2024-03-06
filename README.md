# 矩阵区间压缩

## Quick start

```bash
# https://pypi.org/project/range-compression/
pip install range-compression
```

```python
from range_compression import RangeCompressedMask, mask_encode
from pathlib import Path


mtx = .... # 带有很多连续值的矩阵
rcm = mask_encode(mtx)

X, Y = ..., ... # 要查找的 X, Y
res = rcm.find_index(X, Y)

assert res.shape == X.shape
assert (mtx[Y, X] == res).all()

# 也可以 rcm.save(p)
# 之后在其他地方 rcm = RangeCompressedMask.load(p)
```


## TODO

[] 把性能测试添加到测试和 readme 中
[x] 每个版本做性能回归测试
[ ] 添加更多说明和直接能运行的快速入门

## 关于 Python 版本

支持 >=3.8, <=3.12

但 numba0.59 已经不再支持 Python3.8，如果以后使用到了 numba 新版本的特性，那么 Python3.8 可能不再会支持
