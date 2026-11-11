# 多位复核员资料标签的潜在真值与混淆估计

这是现场机器人应用分析的合成资料项目。`data`保存业务记录，`analysis_data.io`负责严格JSON读取和原始字节指纹，`analysis_data.domain`提供已有的基础汇总。字段与数学口径见[资料约定](docs/data-contract.md)。所有厂区、设备、收益与结果均为合成，不代表任何实际项目的效果。

Python 3.11及以上版本可直接运行，当前无第三方Python依赖。项目根目录中的常用命令：

```sh
python -m unittest discover -s tests -v
python -m compileall -q analysis_data tests
python -m analysis_data.cli inspect data/normal.json
python -m analysis_data.cli fit data/normal.json
```

`inspect`展示原始资料摘要与基础统计；`fit`运行多初值Dawid-Skene MAP-EM，导出每item潜在类别后验与熵、每reviewer混淆矩阵及二部图连通分量；`python -m analysis_data.cli objective data/normal.json 参数.json`用给定参数独立重算目标对数后验，参数文件可直接取fit输出中某个final。推断出的潜在类别不是外部真值，不得回灌计算复核员准确率。结果写入其他目录时应保留输入来源指纹，原始资料不覆盖。更多边界资料与normal采用同样的JSON结构，invalid前缀文件保存不合法输入。

容器可执行 `docker build -t field-analysis .` 后用 `docker run --rm field-analysis` 查看资料，或 `docker run --rm field-analysis python -m unittest discover -s tests -v` 运行现有检查。运行期间无外网或账号要求。
