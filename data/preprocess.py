class DataPreprocessor:
    types = ['fr_en', 'ja_en', 'zh_en']
    sourceKG = {
        'before': ['attrs_1', 'ent_ids_1'],
        'after':  ['att_triple_1', 'att2id_1', 'att_val2id_1']
    }
    targetKG = {
        'before': ['attrs_2', 'ent_ids_2'],
        'after':  ['att_triple_2', 'att2id_2', 'att_val2id_2']
    }
    after_all = ['att_triple_all', 'att2id_all', 'att_value2id_all']
    ent2id = {}
    att2id = {}
    att_val2id = {}
    att_triple2id = []
    att_id = 0  # default if not exist
    att_val_id = 0  # default if not exist

    def getFile(self, types, filename):
        return './' + types + '/' + filename

    def reset(self):
        self.ent2id = {}
        self.att2id = {}
        self.att_val2id = {}
        self.att_triple2id = []

    def process(self):
        self.transform()
        self.merge_all()
        return self

    def transform(self):
        KGS = [self.sourceKG, self.targetKG]
        for t in self.types:
            for kg in KGS:
                before = kg['before']
                self.read(self.getFile(t, before[0]), self.getFile(t, before[1]))
                # './data/ja_en/atts_2', './data/ja_en/ent_ids_2'

                after = kg['after']
                self.save_att_triple(self.getFile(t, after[0]))
                self.save_att2id(self.getFile(t, after[1]))
                self.save_att_val2id(self.getFile(t, after[2]))
                self.reset()
            self.att_id = 0  # default if not exist
            self.att_val_id = 0

    def read(self, triple_datapath='./data/ja_en/atts_2', ent2id_datapath='./data/ja_en/ent_ids_2'):
        # 实体、属性值、属性
        with open(ent2id_datapath, 'r') as r:
            ent2ids = r.readlines()
            print(ent2id_datapath, "\t", len(ent2ids))
        for i in ent2ids:
            ent_id, ent = i.strip().split('\t', 1)
            self.ent2id[ent] = ent_id

        with open(triple_datapath, 'r') as r:
            lines = r.readlines()
            print(triple_datapath, "\t", len(lines))

        for line in lines:
            line = line.strip()
            line = line.replace('<', '')
            line = line.replace('>', '')
            ent, att, att_val = line.split(' ', 2)  # 实体 属性名 属性值
            if att not in self.att2id.keys():
                self.att2id[att] = self.att_id
                self.att_id += 1
            if att_val not in self.att_val2id.keys():
                self.att_val2id[att_val] = self.att_val_id
                self.att_val_id += 1
            single_triple_id = []
            single_triple_id.append(str(self.ent2id[ent]))
            single_triple_id.append(str(self.att_val2id[att_val]))
            single_triple_id.append(str(self.att2id[att]))
            d = '\t'.join(single_triple_id)
            self.att_triple2id.append(d)

    def save_att_triple(self, filepath='./data/ja_en/att_triple_2'):
        print(filepath, "\t", len(self.att_triple2id))
        with open(filepath, 'w') as w:
            for i in self.att_triple2id:
                w.write(i+'\n')

    def save_att2id(self, filepath='./data/ja_en/att2id_2'):
        print(filepath, "\t", len(self.att2id))
        with open(filepath, 'w') as w:
            for i in self.att2id:
                w.write(str(self.att2id[i])+'\t'+i+'\n')

    def save_att_val2id(self, filepath='./data/ja_en/att_val2id_2'):
        print(filepath, "\t", len(self.att_val2id))
        with open(filepath, 'w') as w:
            for i in self.att_val2id:
                w.write(str(self.att_val2id[i])+'\t'+i+'\n')

    def merge_all(self):
        source_after = self.sourceKG['after']
        target_after = self.targetKG['after']
        for t in self.types:
            for i in range(3):
                a = self.getFile(t, source_after[i])
                b = self.getFile(t, target_after[i])
                c = self.getFile(t, self.after_all[i])
                self.merge(a, b, c)
            a = self.getFile(t, 'ent_ids_1')
            b = self.getFile(t, 'ent_ids_2')
            c = self.getFile(t, 'ent_ids_all')
            self.merge(a, b, c)

    def merge(self, a, b, c):
        # a+b -> c
        # ent_ids_1 + ent_ids_2 -> ent_ids_all
        with open(c, 'w') as f_all:
            with open(a, 'r') as f_1:
                f_all.writelines(f_1.readlines())
            with open(b, 'r') as f_2:
                f_all.writelines(f_2.readlines())

    def report(self):
        print(self.types)
        source_after = self.sourceKG['after']
        target_after = self.targetKG['after']
        for t in self.types:
            for i in range(3):
                a = self.getFile(t, source_after[i])
                b = self.getFile(t, target_after[i])
                c = self.getFile(t, self.after_all[i])
                self.report_lines(a)
                self.report_lines(b)
                self.report_lines(c)
            a = self.getFile(t, 'ent_ids_1')
            b = self.getFile(t, 'ent_ids_2')
            c = self.getFile(t, 'ent_ids_all')
            self.report_lines(a)
            self.report_lines(b)
            self.report_lines(c)

    def report_lines(self, path):
        with open(path, "r") as f:
            print(path, "\t\t", len(f.readlines()))



DataPreprocessor().process()

# open atts_2 ent_ids_2
# write att_triple_2 att2id_2 att_val2id_2
