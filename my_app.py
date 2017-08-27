#!/usr/bin/python
# -*- coding: utf-8 -*-
from flask import Flask
import matplotlib; matplotlib.use('Agg')
import matplotlib.pylab as plt
from matplotlib.patches import FancyArrowPatch
import networkx as nx
import numpy as np
import twitter
import os
import json
import constellations

app = Flask(__name__)

GRAPH_FILENAME = '/tmp/graph.png'

auth = twitter.OAuth(
    os.getenv('TWITTER_TOKEN'),
    os.getenv('TWITTER_TOKENSECRET'),
    os.getenv('TWITTER_CONSUMERTOKEN'),
    os.getenv('TWITTER_CONSUMERSECRET')
)
twitter_api = twitter.Twitter(auth=auth)
twitter_upload = twitter.Twitter(domain='upload.twitter.com', auth=auth)

def make_digraph(graph_dict):
    uG = nx.from_dict_of_lists(graph_dict)
    seed_node = np.random.choice(uG.nodes())
    dfs = nx.depth_first_search.dfs_tree(uG, seed_node)
    topological_order = nx.topological_sort(dfs)
    
    dG = nx.DiGraph()

    for edge in uG.edges():
        A, B = edge

        if topological_order.index(A) > topological_order.index(B):
            dG.add_edge(A,B)
        else:
            dG.add_edge(B,A)
            
    return dG, topological_order

def make_computation_graph(dG, topological_order):
    cG = dG.copy()
    
    input_nodes = [n for n in cG.nodes() if cG.in_degree()[n] == 0]
    output_nodes = [n for n in cG.nodes() if cG.out_degree()[n] == 0]
    labels = {}

    for node in cG.nodes():
        if node in input_nodes:
            cG.node[node]['OP'] = 'INPUT'
            cG.node[node]['STATE'] = np.random.randint(0,19)
        elif cG.in_degree()[node] > 1:
            cG.node[node]['OP'] = np.random.choice([u'Σ','*'])
        else:
            OP = np.random.choice(
                ['SIN', 'COS', '+']
            )
            cG.node[node]['OP'] = OP
            if OP == '+':
                cG.node[node]['CONSTANT'] = 100*(np.random.rand() - 0.5)

    for n in reversed(topological_order):

        node = cG.node[n]
        OP = node['OP']

        if OP == 'INPUT':
            labels[n] = node['STATE']
            continue

        if cG.in_degree()[n] > 1:        
            inputs = [cG.node[pre]['STATE'] for pre in cG.predecessors(n)]
            if OP == u'Σ':
                node['STATE'] = np.sum(inputs)
            elif OP == '*':
                node['STATE'] = np.prod(inputs)

        else:
            pre = cG.predecessors(n)[0]
            input = cG.node[pre]['STATE']

            if OP == 'SIN':
                node['STATE'] = np.sin(input)
            elif OP == 'COS':
                node['STATE'] = np.cos(input)
            elif OP == '+':
                node['STATE'] = input + node['CONSTANT']

            elif OP == 'EXP':
                node['STATE'] = np.exp(input)
            elif OP == 'LOG':
                node['STATE'] = np.log(input)

        labels[n] = OP

    node_colors = ['g' if n in input_nodes else 'b' if n in output_nodes else 0.8 for n in cG.nodes()]
    
    return cG, labels, node_colors

def draw_graph(cG, layout, labels, node_colors):
    fig = plt.figure(figsize=(10,10))

    nx.draw(
        cG,
        layout,
        with_labels=True,
        labels=labels,
        arrows=False,
        node_size=800,
        node_color=node_colors
    )

    ax = plt.gca()
    p=0.75
    
    for A, B in cG.edges():
        src, dst = layout[A], layout[B]

        x1, y1 = src
        x2, y2 = dst
        dx = x2-x1   # x offset
        dy = y2-y1   # y offset
        d = np.sqrt(float(dx**2 + dy**2))  # length of edge
        if d == 0:   # source and target at same position
            continue
        if dx == 0:  # vertical edge
            xa = x2
            ya = dy*p+y1
        if dy == 0:  # horizontal edge
            ya = y2
            xa = dx*p+x1
        else:
            theta = np.arctan2(dy, dx)
            xa = p*d*np.cos(theta)+x1
            ya = p*d*np.sin(theta)+y1


        arrows = FancyArrowPatch(posA=(x1, y1), posB=(xa, ya),
                                color = 'k',
                                arrowstyle="-|>",
                                mutation_scale=1000**.5,
                                zorder=0,
                                connectionstyle="arc3")

        ax.add_patch(arrows)


def render_function(cG, const_name):
    input_nodes = [n for n in cG.nodes() if cG.in_degree()[n] == 0]
    output_nodes = [n for n in cG.nodes() if cG.out_degree()[n] == 0]

    
    in_str = ','.join([ str(cG.node[node]['STATE']) for node in input_nodes])
    out_str = ','.join([ "{0:.1f}".format(cG.node[node]['STATE']) for node in output_nodes])

    return ('%s; f(%s) = %s' % (const_name, in_str, out_str))


@app.route('/')
def post():
    const_name = np.random.choice(list(constellations.GRAPHS.keys()))

    dG, topological_order = make_digraph(constellations.GRAPHS[const_name])
    cG, labels, node_colors = make_computation_graph(dG, topological_order)
    draw_graph(cG, constellations.LAYOUTS[const_name], labels, node_colors)
    plt.savefig(GRAPH_FILENAME, dpi=600)

    with open(GRAPH_FILENAME, "rb") as f:
        image_data = f.read()

    image_id = twitter_upload.media.upload(media=image_data)["media_id_string"]

    res = twitter_api.statuses.update(status=render_function(cG, const_name), media_ids=image_id)
    return json.dumps(res)


if __name__ == '__main__':
    app.run()
