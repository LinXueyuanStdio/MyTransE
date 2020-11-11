with open("fr_en/triples_struct_all") as f:
    lines = f.readlines()
    hs = []
    rs = []
    ts = []
    for i in lines:
        isp = i.split("\t")
        h, r, t = int(isp[0]), int(isp[1]), int(isp[2])
        hs.append(h)
        rs.append(r)
        ts.append(t)
    hs = list(set(hs))
    rs = list(set(rs))
    ts = list(set(ts))
    print(max(hs), len(hs))
    print(max(rs), len(rs))
    print(max(ts), len(ts))
