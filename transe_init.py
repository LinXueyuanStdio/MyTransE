import torch
import torch.nn as nn


def read_ids_and_names(dir_path="data/fr_en/ent_ids_all", sp="\t"):
    ids = []
    names = []
    with open(dir_path, encoding="utf-8") as file:
        lines = file.readlines()
        for line in lines:
            id_to_name = line.strip().split(sp)
            ids.append(int(id_to_name[0]))
            names.append(id_to_name[1])
    return ids, names


def read_align(entity_align_file="data/fr_en/ref_ent_ids"):
    ret = []
    with open(entity_align_file, encoding='utf-8') as f:
        for line in f:
            th = line[:-1].split('\t')
            ret.append((int(th[0]), int(th[1])))
    return ret


e, _ = read_ids_and_names()
nentity = len(e)
print(nentity)
seeds = read_align()

entity_weight = torch.zeros(nentity, 200)
nn.init.normal_(
    tensor=entity_weight,
    mean=0,
    std=1
)
for left_entity, right_entity in seeds:
    entity_weight[left_entity] = entity_weight[right_entity]

with open("./transe_init.txt", "w") as f:
    w = entity_weight.view(-1)
    for i in w:
        f.write(str(i.item()) + "\n")