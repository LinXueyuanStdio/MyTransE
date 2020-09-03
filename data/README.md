# DBP15k 属性数据集

attrs_1, attrs_2 下载地址：http://ws.nju.edu.cn/jape/

| 文件名           | schame                                       | 中文描述                       |
| :--------------- | :------------------------------------------- | :---------------------------- |
| ent_ids_1        | (entity_id1, entity_name1)                   | 源知识图谱的实体id（id化）      |
| ent_ids_2        | (entity_id2, entity_name2)                   | 目标知识图谱的实体id（id化）    |
| attrs_1          | (entity_name1, attr_name1, attr_value_name1) | 源知识图谱的属性三元组          |
| attrs_2          | (entity_name2, attr_name2, attr_value_name2) | 目标知识图谱的属性三元组        |
| ent_ids_all      | (entity_id1, entity_name2)                   | 源KG+目的KG的实体（id化）       |
| att_triple_all   | (entity_id, attr_id, attr_value_id)          | 源KG+目的KG的属性三元组（id化） |
| att2id_all       | (attr_id, attr_name)                         | 源KG+目的KG的属性（id化）       |
| att_value2id_all | (attr_value_id, attr_value_name)             | 源KG+目的KG的属性值（id化）     |

## 运行数据预处理

```shell
python preprocess.py
```

将会读取 4 个文件（ent_ids_1,ent_ids_2,attrs_1,attrs_2）

生成训练模型要用到的（ent_ids_all，att_triple_all，att2id_all，att_value2id_all）
