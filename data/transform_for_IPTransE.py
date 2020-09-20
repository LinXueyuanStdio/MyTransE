
triple_filename = "triple2id.txt"
align_filename = "common_entities2id.txt"  # 左实体

lang = "fr_en"

source_triple_filename = "att_triple_all"
source_align_filename = "ref_ent_ids"

align_entities = {}
with open(lang+"/"+source_align_filename, "r") as source_file:
    with open(lang+"/"+align_filename, "w") as target_file:
        lines = source_file.readlines()
        max_lines = len(lines)
        training_lines = int(0.3*max_lines)
        for i in range(max_lines):
            line = lines[i]
            [e1, e2] = line.strip().split("\t")
            if i < training_lines:
              align_entities[e1] = e2
              align_entities[e2] = e1
              target_file.write(str(e1)+"\n")
            else:
              break

triple_count = 0
with open(lang+"/"+source_triple_filename, "r") as source_file:
    with open(lang+"/"+triple_filename, "w") as target_file:
        for line in source_file.readlines():
            [e, v, a] = line.strip().split("\t")
            target_file.write(str(e)+"\n"+str(v)+"\n"+str(a)+"\n")
            triple_count += 1
            if e in align_entities:
              target_file.write(str(align_entities[e])+"\n"+str(v)+"\n"+str(a)+"\n")
              triple_count += 1

print(triple_count)
