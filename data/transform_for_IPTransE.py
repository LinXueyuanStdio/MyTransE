
triple_filename = "triple2id.txt"

align_filename = "common_entities2id.txt"  # 左实体

lang = "fr_en"

source_triple_filename = "att_triple_all"

source_align_filename = "ref_ent_ids"

with open(lang+"/"+source_triple_filename, "r") as source_file:
    with open(lang+"/"+triple_filename, "w") as target_file:
        for line in source_file.readlines():
            [e, v, a] = line.split("\t")
            target_file.write(str(e)+"\n"+str(v)+"\n"+str(a))

with open(lang+"/"+source_align_filename, "r") as source_file:
    with open(lang+"/"+align_filename, "w") as target_file:
        for line in source_file.readlines():
            [e1, _] = line.split("\t")
            target_file.write(str(e1)+"\n")
