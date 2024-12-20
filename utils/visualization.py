import mlflow
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

from tot.thought_node import ThoughtNode
from tot.tree_of_thoughts import ToT
from utils.processing import preprocess_text_for_visualization

def generate_tree_visualization(tot: ToT, display=True, title="Tree of Thoughts", logging_id="."):
    """
    Visualizes the Tree of Thoughts.
    """
    graph = nx.DiGraph()

    for node in tot.all_nodes:
        graph.add_node(
            node.id,
            thought=preprocess_text_for_visualization(node.thought),
            level=node.tree_level,
            resources=node.next_step_allocated_resources,
            quality=node.quality_score,
            efficiency=node.efficiency_score,
            reward=node.combined_reward,
            cumulative_reward=node.get_cumulative_reward(),
            winner=node.winner
        )
        if node.parent:
            graph.add_edge(node.parent.id, node.id)

    pos = {}
    layer_separation = 1.0
    node_separation = 0.5

    for node in graph.nodes(data=True):
        level = node[1]['level']
        siblings_at_level = [n for n in graph.nodes(data=True) if n[1]['level'] == level]
        sibling_index = siblings_at_level.index(node)
        num_siblings = len(siblings_at_level)

        x = sibling_index * node_separation - (num_siblings * node_separation) / 2
        y = -level * layer_separation
        pos[node[0]] = (x, y)

    # Red to green colormap
    cmap = LinearSegmentedColormap.from_list("rg", ["r", "y", "g"], N=256)

    # Normalize rewards for color mapping
    rewards = [node[1].get('reward', 0) for node in graph.nodes(data=True)]
    if rewards and max(rewards) != min(rewards):
        norm_rewards = [(r - min(rewards)) / (max(rewards) - min(rewards)) for r in rewards]
    else:
        norm_rewards = rewards

    # Plotting graph
    plt.rc('text', usetex=False)
    fig, ax = plt.subplots(figsize=(12, 8))
    nodes = nx.draw_networkx_nodes(graph, pos, node_size=500, node_color=norm_rewards, cmap=cmap, ax=ax)

    # Winner colored edges
    winner_edges = [(u, v) for u, v, data in graph.edges(data=True) if graph.nodes[v]['winner']]
    non_winner_edges = [(u, v) for u, v, data in graph.edges(data=True) if not graph.nodes[v]['winner']]
    nx.draw_networkx_edges(graph, pos, edgelist=non_winner_edges, arrows=True, ax=ax)
    nx.draw_networkx_edges(graph, pos, edgelist=winner_edges, arrows=True, edge_color="blue", width=2, ax=ax)

    # Labels and tooltips
    labels = {}
    tooltips = {}
    for node_id, data in graph.nodes(data=True):
        if data:
            thought = data['thought'] if data['thought'] else "N/A"
            label = f"{node_id}\n{thought[:20]}"
            labels[node_id] = label
            tooltip = f"ID: {node_id}\nThought: {data['thought']}\nLevel: {data['level']}\nResources: {data.get('resources', 0)}\nQS: {data.get('quality', 0):.2f}\nES: {data.get('efficiency', 0):.2f}\nReward: {data.get('reward', 0):.2f}\nCumulative Reward: {data.get('cumulative_reward', 0):.2f}"
            tooltips[node_id] = tooltip

    nx.draw_networkx_labels(graph, pos, labels, font_size=8, ax=ax)
    
    import mplcursors
    def on_add(sel):
        x, y = sel.target
        distances = {node_id: np.sqrt((x - pos[node_id][0])**2 + (y - pos[node_id][1])**2) for node_id in pos}
        closest_node_id = min(distances, key=distances.get) #gets the key of the minimum distance
        sel.annotation.set_text(tooltips[closest_node_id])
    
    mplcursors.cursor(nodes).connect("add", on_add)
    
    plt.colorbar(plt.cm.ScalarMappable(cmap=cmap), label="Quality Score", ax=ax)
    
    ax.set_title(title)
    ax.axis("off")

    mlflow.log_figure(fig, f"{logging_id}/visuals/tree_visualization_{tot.node_count}.png")

    if display:
        plt.show()
    else:
        plt.close()