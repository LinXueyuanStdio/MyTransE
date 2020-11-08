files = ["triples_1", "triples_2"]
lang = ["fr_en", "ja_en", "zh_en"]
result = "triples_struct_all"
for i in lang:
    f0 = open(i + "/" + files[0])
    f1 = open(i + "/" + files[1])
    f2 = open(i + "/" + result, "w")
    f2.writelines(f0.readlines())
    f2.writelines(f1.readlines())
    f0.close()
    f1.close()
    f2.close()
