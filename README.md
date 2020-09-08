# MyTransE

目标：复现 TransE 并做属性消融实验

## 说明

- main.py 是参考 RotatE 的 TransE 实现，仅在测试时用到对齐数据。
- main2.py 是我根据TransE原始论文另一个实现，用于验证模型有效性。
- main3.py 是在 main.py 的基础上，将对齐数据划分80%用于初始化，20%用于测试。
  - 初始化：对于对齐的实体 (a,b)，先使用正态分布初始化所有实体的嵌入，再用 b 的嵌入覆盖 a 的嵌入，使得实体在初始状态时共享相同的嵌入。
- main4.py 是在 main3.py 的基础上，对数据集做了数据增强。
  - 数据增强：对于训练集的对齐实体 (a,b)，原属性三元组(a,attr,value)，增加三元组(b,attr,value)。

## 实验结果

具体训练结果在日志文件中 `main_log.txt` 和 `main3_log.txt`。
- `main_log.txt`：使用属性三元组训练模型，将对齐数据用于测试。
- `main3_log.txt`：使用对齐数据80%用于初始化，20%用于测试；使用属性三元组训练模型。训练过程没有用到对齐数据。

属性消融实验，以 fr_en 为例

- 未使用对齐数据
  ```
  For each left:
  Hits@1: 0.01%
  Hits@10: 0.05%
  Hits@50: 0.32%
  Hits@100: 0.61%
  For each right:
  Hits@1: 0.00%
  Hits@10: 0.10%
  Hits@50: 0.35%
  Hits@100: 0.68%
  ```
- 使用对齐数据80%用于训练，20%用于测试
  ```
  For each left:
  Hits@1: 0.30%
  Hits@10: 1.77%
  Hits@50: 6.20%
  Hits@100: 10.47%
  For each right:
  Hits@1: 0.13%
  Hits@10: 2.20%
  Hits@50: 7.84%
  Hits@100: 11.70%
  ```