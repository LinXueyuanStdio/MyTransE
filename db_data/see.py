def see(filename):
    entity_ids = []
    relation_ids = []
    triples = []
    f = open(filename)
    for l in f.readlines():
        h, r, t = [int(j) for j in l.split("\t")]
        entity_ids.append(h)
        entity_ids.append(t)
        relation_ids.append(r)
        triples.append((h, r, t))
    f.close()
    entity_ids = list(set(entity_ids))
    sorted(entity_ids)
    relation_ids = list(set(relation_ids))
    print("实体个数", len(entity_ids))
    print("关系个数", len(relation_ids))
    # print(entity_ids)
    return triples, entity_ids, relation_ids


def read_seed(filename):
    entity_ids1 = []
    entity_ids2 = []
    seeds = []
    f = open(filename)
    for l in f.readlines():
        e1, e2 = [int(j) for j in l.split("\t")]
        entity_ids1.append(e1)
        entity_ids2.append(e2)
        seeds.append((e1, e2))
    f.close()
    return seeds, entity_ids1, entity_ids2


"ref_ent_ids"
"ent_ids_all"
"att2id_all"
"att_value2id_all"
"triples_struct_all"
"ent_ids_1"
"ent_ids_2"
triples1, entity_ids1, relation_ids1 = see("train1.txt")
triples2, entity_ids2, relation_ids2 = see("train2.txt")
seeds, _, _ = read_seed("seeds.txt")

entity_ids1_count = len(entity_ids1)
offset = entity_ids1_count
with open("ent_ids_1", "w") as f:
    for e in entity_ids1:
        f.write("%d\t%s\n" % (e, "kg1_name"))
with open("ent_ids_2", "w") as f:
    for e in entity_ids2:
        f.write("%d\t%s\n" % (e + offset, "kg2_name"))
with open("ent_ids_all", "w") as f:
    f2 = open("att_value2id_all", "w")
    for e in entity_ids1:
        f.write("%d\t%s\n" % (e, "kg1_name"))
        f2.write("%d\t%s\n" % (e, "kg1_name"))
    for e in entity_ids2:
        f.write("%d\t%s\n" % (e + offset, "kg2_name"))
        f2.write("%d\t%s\n" % (e + offset, "kg2_name"))
with open("ref_ent_ids", "w") as f:
    for e1, e2 in seeds:
        f.write("%d\t%d\n" % (e1, e2 + offset))
with open("att2id_all", "w") as f:
    for r in relation_ids1:
        f.write("%d\t%s\n" % (r, "relation_name"))
with open("triples_struct_all", "w") as f:
    for h, r, t in triples1:
        f.write("%d\t%d\t%d\n" % (h, r, t))
    for h, r, t in triples2:
        f.write("%d\t%d\t%d\n" % (h + offset, r, t + offset))
