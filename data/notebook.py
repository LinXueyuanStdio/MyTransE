lang = "fr_en"

source_triple_filename = "att_triple_all"
source_align_filename = "ref_ent_ids"

ent_set = set()
with open(lang + "/" + source_align_filename, "r") as source_file:
    lines = source_file.readlines()
    max_lines = len(lines)
    for i in range(max_lines):
        line = lines[i]
        [e1, e2] = line.strip().split("\t")
        ent_set.add(int(e1))
        ent_set.add(int(e2))

triple_count = 0
with open(lang + "/" + source_triple_filename, "r") as source_file:
    size = 0
    for line in source_file.readlines():
        [e, v, a] = line.strip().split("\t")
        if int(e) in ent_set:
            triple_count += 1
        size += 1
    print(size)
    print(triple_count)
