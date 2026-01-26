inp = "datasets/vertebrate_mammalian_NM_human_mRNA_filtered_separators.fasta"
out = "datasets/raw.txt"

with open(inp) as fin, open(out, "w") as fout:
    buf = []
    for line in fin:
        if line.startswith(">"):
            if buf:
                fout.write("".join(buf).upper() + "\n")
                buf = []
        else:
            buf.append(line.strip())
    if buf:
        fout.write("".join(buf).upper() + "\n")