import dgl
import os

def construct_graph(path):
    paper_ids = []
    paper_names = []
    author_ids = []
    author_names = []
    conf_ids = []
    conf_names = []
    f_3 = open(os.path.join(path, "id_author.txt"), encoding="ISO-8859-1")
    f_4 = open(os.path.join(path, "id_conf.txt"), encoding="ISO-8859-1")
    f_5 = open(os.path.join(path, "paper.txt"), encoding="ISO-8859-1")
    while True:
        z = f_3.readline()
        if not z:
            break
        z = z.strip().split()
        identity = int(z[0])
        author_ids.append(identity)
        author_names.append(z[1])
    while True:
        w = f_4.readline()
        if not w:
            break;
        w = w.strip().split()
        identity = int(w[0])
        conf_ids.append(identity)
        conf_names.append(w[1])
    while True:
        v = f_5.readline()
        if not v:
            break;
        v = v.strip().split()
        identity = int(v[0])
        paper_name = 'p' + ''.join(v[1:])
        paper_ids.append(identity)
        paper_names.append(paper_name)
    f_3.close()
    f_4.close()
    f_5.close()

    author_ids_invmap = {x: i for i, x in enumerate(author_ids)}
    conf_ids_invmap = {x: i for i, x in enumerate(conf_ids)}
    paper_ids_invmap = {x: i for i, x in enumerate(paper_ids)}

    paper_author_src = []
    paper_author_dst = []
    paper_conf_src = []
    paper_conf_dst = []
    f_1 = open(os.path.join(path, "paper_author.txt"), "r")
    f_2 = open(os.path.join(path, "paper_conf.txt"), "r")
    for x in f_1:
        x = x.split('\t')
        x[0] = int(x[0])
        x[1] = int(x[1].strip('\n'))
        paper_author_src.append(paper_ids_invmap[x[0]])
        paper_author_dst.append(author_ids_invmap[x[1]])
    for y in f_2:
        y = y.split('\t')
        y[0] = int(y[0])
        y[1] = int(y[1].strip('\n'))
        paper_conf_src.append(paper_ids_invmap[y[0]])
        paper_conf_dst.append(conf_ids_invmap[y[1]])
    f_1.close()
    f_2.close()

    pa = dgl.bipartite((paper_author_src, paper_author_dst), 'paper', 'pa', 'author')
    ap = dgl.bipartite((paper_author_dst, paper_author_src), 'author', 'ap', 'paper')
    pc = dgl.bipartite((paper_conf_src, paper_conf_dst), 'paper', 'pc', 'conf')
    cp = dgl.bipartite((paper_conf_dst, paper_conf_src), 'conf', 'cp', 'paper')
    hg = dgl.hetero_from_relations([pa, ap, pc, cp])
    return hg, author_names, conf_names, paper_names





def gen_metapath(hg, start_node_type, num_samples_per_node, metapath, walk_length):
    traces = []
    for _ in range(num_samples_per_node):
        trace, _ = dgl.sampling.random_walk(hg, hg.nodes(start_node_type), metapath=metapath*walk_length)
        traces.extend(trace)

    return traces

if __name__ == '__main__':
    metapath = ['cp', 'pa', 'ap', 'pc']
