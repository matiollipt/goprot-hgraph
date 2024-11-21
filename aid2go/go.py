# Standard library imports
import math
import os
import pickle
import random
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import textwrap

# Third-party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colormaps
import requests
from tqdm.auto import tqdm
import networkx as nx

# Machine learning and data processing
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, silhouette_samples
import hdbscan

# Bioinformatics and deep learning
from obonet import read_obo
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, HeteroData
from transformers import BertTokenizer, BertModel


def get_godag(save_path="./data/go", timeout=30):
    """
    Downloads the basic Gene Ontology (GO) graph in OBO format from the provided URL,
    saves it locally, and loads it into a NetworkX DiGraph object.
    It also creates a pandas DataFrame from the GO graph nodes.

    Parameters:
    save_path (str): The path where the downloaded file and the saved DataFrame will be saved.
    timeout (int): The maximum number of seconds to wait for a response from the server.

    Returns:
    go_dag (NetworkX DiGraph): The loaded GO graph.
    go_df (pandas DataFrame): The DataFrame created from the GO graph nodes.
    """
    # Define local filepath
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    go_filepath = save_path / "go-basic.obo"

    # URL of the GO OBO file
    address = "https://purl.obolibrary.org/obo/go/go-basic.obo"

    try:
        print(f"Attempting to download the GO file from {address}")

        # Download the file with a timeout and handle potential errors
        response = requests.get(address, timeout=timeout)
        response.raise_for_status()  # Raises HTTPError for bad responses (4xx, 5xx)

    except requests.exceptions.Timeout:
        print(f"Error: The request timed out after {timeout} seconds. Exiting...")
        return None, None

    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}. Exiting...")
        return None, None

    except requests.exceptions.ConnectionError:
        print(
            "Error: Failed to establish a connection. Please check your network. Exiting..."
        )
        return None, None

    except requests.exceptions.RequestException as req_err:
        print(f"An error occurred during the request: {req_err}. Exiting...")
        return None, None

    finally:
        # Ensure a graceful exit if the request did not succeed
        print("Exiting the function after request completion.")

    # If download is successful, save the OBO file locally
    print(f"Download successful! Saving GO OBO file to {go_filepath}")
    with open(go_filepath, "wb") as f:
        f.write(response.content)

    # Load GO graph from the local OBO file
    go_dag = read_obo(go_filepath)

    # Create a DataFrame from GO graph nodes
    nodes_list = []
    for go_term, attr in go_dag.nodes.items():
        definition = attr["def"].split('"', 2)[1] if "def" in attr else ""
        nodes_list.append(
            {
                "go_id": go_term,
                "name": attr["name"],
                "namespace": attr["namespace"],
                "definition": definition,
                "def_word_count": len(re.findall(r"\w+", definition)),
            }
        )

    go_df = pd.DataFrame(nodes_list)

    # Check if the number of nodes matches the number of entries in the GO DAG
    assert len(go_df) == len(go_dag.nodes.items()), "Mismatch in the number of nodes."

    # Calculate degree counts
    in_degree_dict = dict(go_dag.in_degree)
    out_degree_dict = dict(go_dag.out_degree)

    # Map degree counts to GO terms
    go_df["in_degree"] = go_df["go_id"].map(in_degree_dict)
    go_df["out_degree"] = go_df["go_id"].map(out_degree_dict)
    go_df["degree"] = go_df["in_degree"] + go_df["out_degree"]

    # Save the GO DataFrame as CSV
    df_filepath = save_path / "go-basic.csv"
    go_df.to_csv(df_filepath, index=False)

    # Print information about the GO DAG and DataFrame
    print(f"GO DAG contains {len(go_dag.nodes)} nodes.")
    print(f"Format version: {go_dag.graph.get('format-version', 'Unknown')}")
    print(f"Data version: {go_dag.graph.get('data-version', 'Unknown')}")
    print(f"GO DAG saved to: {df_filepath}")
    print(f"GO DataFrame shape: {go_df.shape}")

    return go_dag, go_df


def get_go_relations(go_dag, save_path):
    """
    Extracts and organizes "is_a" and other relationships between GO terms from a given directed acyclic graph (DAG).

    Parameters:
    go_dag (networkx.DiGraph): The directed acyclic graph representing the Gene Ontology (GO) terms and their relationships.
    save_path (str): The path where the extracted relationships will be saved.

    Returns:
    go_isa_df (pandas.DataFrame): A DataFrame containing "is_a" relationships between GO terms.
    go_relationships_df (pandas.DataFrame): A DataFrame containing other relationships between GO terms.
    go_edges_df (pandas.DataFrame): A DataFrame containing all relationships between GO terms.
    """

    # Ensures destination directory
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    # Extract "is_a" relationships between GO terms
    columns = ["source_go_id", "relationship", "target_go_id"]

    # Extract other relationships between GO terms
    go_edges = [[i, k, j] for i, j, k in go_dag.edges]

    # Create DataFrame and save
    go_edges_df = pd.DataFrame(data=go_edges, columns=columns)
    go_edges_df.to_csv(save_path / "go_edges.tsv", sep="\t", index=False)

    # Print summary
    print("GO edges df dimensions:", go_edges_df.shape)
    print(go_edges_df["relationship"].value_counts())

    return go_edges_df


def has_key(d, key):
    """
    Check if a given key exists in a nested dictionary.

    Parameters:
    d (dict): The nested dictionary to search in.
    key (str): The key to search for.

    Returns:
    bool: True if the key is found, False otherwise.
    """
    if key in d:
        return True
    for k, v in d.items():
        if isinstance(v, dict):
            if has_key(v, key):
                return True
    return False


class TextDataset(Dataset):
    def __init__(self, dataframe, text_column):
        """Initialize a TextDataset instance."""
        self.texts = dataframe[text_column].tolist()

    def __len__(self):
        """Return the total number of texts in the dataset."""
        return len(self.texts)

    def __getitem__(self, idx):
        """Get a text item from the dataset based on the given index."""
        return self.texts[idx]


def embed_texts(
    df,
    column_id,
    column_text,
    batch_size=32,
    pre_trained_model="bert-base-uncased",
    save_path=None,
):
    """Embeds texts from a DataFrame using a pre-trained BERT model."""

    save_path = Path(save_path)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load pre-trained BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained(pre_trained_model)
    model = BertModel.from_pretrained(pre_trained_model).to(device)

    # Create TextDataset and DataLoader for batch processing of texts
    dataset = TextDataset(df, column_text)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Initialize dict to store embeddings
    embeddings_dict = dict()
    id_text_mapping = df.set_index(column_text)[column_id].to_dict()

    # Embed texts and store embeddings in dictionary
    for texts in tqdm(dataloader, total=len(dataloader)):
        encoded_input = tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt"
        ).to(device)

        # Compute embeddings and store in dictionary  (Note: model is in evaluation mode)
        with torch.no_grad():
            model_output = model(**encoded_input)
            embeddings = model_output.last_hidden_state.mean(dim=1)

            # Detach and move to CPU before saving
            for text, emb in zip(texts, embeddings):
                go_id = id_text_mapping[text]
                embeddings_dict[go_id] = emb.cpu().numpy()

    # Save ID mappings and embeddings
    with open(save_path / f"go_emb_dict-{column_text}.pkl", "wb") as file:
        pickle.dump(embeddings_dict, file, protocol=pickle.HIGHEST_PROTOCOL)

    with open(save_path / f"id_text_mapping-{column_text}.pkl", "wb") as file:
        pickle.dump(id_text_mapping, file, protocol=pickle.HIGHEST_PROTOCOL)

    return embeddings_dict, id_text_mapping


def create_go_data(go_edges_df, go_embed_dict, save_path, multi_edge=False):
    """
    This function creates edge index and edge attribute tensors for a given GO edges DataFrame and a mapping of GO IDs.

    Parameters:
    go_edges_df (pandas.DataFrame): A DataFrame containing GO edges with columns 'source_go_id', 'target_go_id', and 'relationship'.
    go_embed_dict (dict): A dictionary mapping GO IDs to their embedding vectors.
    save_path (Path): Path to save the resulting data object.
    multi_edge (bool): Whether to create a multi-edge HeteroData object (True) or a homogeneous Data object (False).

    Returns:
    data (HeteroData or Data): A PyTorch Geometric data object containing nodes and edges.
    """
    # Obtain GO identifier mapping
    go_ids = list(go_embed_dict.keys())
    go_feats = torch.stack(
        [torch.tensor(v, dtype=torch.float32) for v in go_embed_dict.values()]
    )
    go_ids_map = dict(zip(go_ids, range(len(go_ids))))

    # Create a mapping for relationship types
    relationship_types = list(go_edges_df["relationship"].unique())

    if multi_edge:
        # Create a HeteroData object
        data = HeteroData()

        # Add GO term nodes to the graph
        data["go"].x = go_feats

        for relationship in relationship_types:
            # Filter edges for current relationship
            edges = go_edges_df[go_edges_df["relationship"] == relationship]

            # Map GO IDs to node indices
            source_nodes = [go_ids_map[go_id] for go_id in edges["source_go_id"]]
            target_nodes = [go_ids_map[go_id] for go_id in edges["target_go_id"]]

            # Create edge index tensor
            edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)

            # Add the edge index to the HeteroData object under the appropriate edge type
            data["go", relationship, "go"].edge_index = edge_index

        # Save the HeteroData object
        torch.save(data, save_path / "go_data_multiedge.pt")

        # Print information about the data
        print("Heterogeneous Graph Information:")
        print(f"Number of nodes: {data['go'].x.size(0)}")
        print("Data structure:")
        print(data)

        return data

    else:
        # Initialize lists for edge indices and edge attributes
        edge_index = []
        edge_attr = []

        relationship_to_index = {rel: i for i, rel in enumerate(relationship_types)}

        # Populate edge_index and edge_attr lists
        for _, row in go_edges_df.iterrows():
            source = go_ids_map[row["source_go_id"]]
            target = go_ids_map[row["target_go_id"]]
            rel_index = relationship_to_index[row["relationship"]]

            edge_index.append([source, target])
            edge_attr.append(rel_index)

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.long)

        # Create a Data object
        data = Data(x=go_feats, edge_index=edge_index, edge_attr=edge_attr)

        # Save the Data object
        torch.save(data, save_path / "go_data.pt")

        # Print information about the data
        print("Homogeneous Graph Information:")
        print(f"Number of nodes: {data.num_nodes}")
        print(f"Number of edges: {data.num_edges}")
        print(f"Edge index shape: {data.edge_index.shape}")
        print(f"Relationship types: {relationship_types}")
        print("Data structure:")
        print(data)

        return data


def analyze_clusters(
    embeddings,
    labels,
    num_of_neighbors,
    save_path: Path,
    silhouette_threshold: float,
    cmap="nipy_spectral",
    show_plot=False,
):
    """
    Analyzes clusters for a given sample using HDBSCAN with various min_cluster_size values,
    saves the plots to the specified path, and returns the data points with silhouette score
    above and below the specified threshold.

    Parameters:
    embeddings (np.ndarray): The sample data for clustering.
    labels (list): List of labels corresponding to the data points in embeddings.
    num_of_neighbors (list): List of min_cluster_size values to try.
    save_path (Path): Path object where plots will be saved.
    silhouette_threshold (float): The threshold for silhouette scores to filter data points.

    Returns:
    dict: Contains results for each min_cluster_size, with data points above and below the silhouette score threshold.
    """
    results = {}

    for n in num_of_neighbors:
        high_silhouette_data_points = []
        low_silhouette_data_points = []

        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        # Initialize the HDBSCAN clusterer
        clusterer = hdbscan.HDBSCAN(min_cluster_size=n)
        cluster_labels = clusterer.fit_predict(embeddings)

        # Filter out noise points (cluster_label == -1)
        filtered_indices = cluster_labels != -1
        filtered_cluster_labels = cluster_labels[filtered_indices]

        # Filter embeddings and labels
        labels = np.array(labels)
        filtered_X = embeddings[filtered_indices]
        filtered_labels = labels[filtered_indices]
        noisy_X = embeddings[~filtered_indices]
        noisy_labels = labels[~filtered_indices]

        if len(filtered_cluster_labels) == 0:
            print(f"# neighbors = {n}, all points are considered noise.")
            continue

        # The silhouette_score gives the average value for all the samples.
        silhouette_avg = silhouette_score(filtered_X, filtered_cluster_labels)
        print(f"# neighbors = {n}")
        print(f"# clusters = {len(np.unique(filtered_cluster_labels))}")
        print(f"Mean Silhouette Score = {silhouette_avg:.2f}")
        print(f"# clustered datapoints = {filtered_cluster_labels.shape[0]}")
        print(f"# noisy datapoints = {noisy_labels.shape[0]}\n")

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(
            filtered_X, filtered_cluster_labels
        )

        y_lower = 10
        for i in range(len(np.unique(filtered_cluster_labels))):
            # Aggregate the silhouette scores for samples belonging to cluster i, and sort them
            ith_cluster_silhouette_values = sample_silhouette_values[
                filtered_cluster_labels == i
            ]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            # Set color for each cluster
            color = colormaps.get_cmap(cmap)(
                i / (len(np.unique(filtered_cluster_labels)) - 1)
            )

            ax1.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_silhouette_values,
                color=color,  # Use color instead of facecolor and edgecolor
                alpha=0.7,
            )

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("Silhouette Plot (Clusters)")
        ax1.set_xlabel("Silhouette Coefficient Values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2nd Plot showing the actual clusters formed
        ax2.scatter(
            filtered_X[:, 0],
            filtered_X[:, 1],
            marker=".",
            s=30,
            lw=0,
            alpha=0.7,
            c=filtered_cluster_labels,  # Use cluster labels to color the clusters
            cmap=colormaps.get_cmap(cmap),  # Specify the colormap
            edgecolor="k",
        )

        # Labeling the clusters
        centers = clusterer.exemplars_
        if centers is not None:
            centers = np.array([np.mean(center, axis=0) for center in centers])
            # Draw white circles at cluster centers
            ax2.scatter(
                centers[:, 0],
                centers[:, 1],
                marker="o",
                c="white",
                alpha=1,
                s=200,
                edgecolor="k",
            )

            for i, c in enumerate(centers):
                ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

        ax2.set_title("Clustered Data Visualization")
        ax2.set_xlabel("Feature space (1st feature)")
        ax2.set_ylabel("Feature space (2nd feature)")

        plt.suptitle(
            "Silhouette analysis for HDBSCAN clustering on sample data with min_cluster_size = %d"
            % n,
            fontsize=14,
            fontweight="bold",
        )

        # Save the plot
        save_file_path = save_path / f"cluster_min_cluster_{n}.png"
        plt.savefig(save_file_path, dpi=300)

        if show_plot:
            plt.show()

        plt.close()

        # Get the data points with silhouette score above and below the specified threshold
        for point, label, silhouette, go_term in zip(
            filtered_X,
            filtered_cluster_labels,
            sample_silhouette_values,
            filtered_labels,
        ):
            data_point_info = {
                "data_point": point,
                "cluster_label": label,
                "silhouette_score": silhouette,
                "go_term": go_term,
            }
            if silhouette > silhouette_threshold:
                high_silhouette_data_points.append(data_point_info)
            else:
                low_silhouette_data_points.append(data_point_info)

        results[n] = {
            "above_threshold": high_silhouette_data_points,
            "below_threshold": low_silhouette_data_points,
        }

    return results


def plot_go_term(go_dag, go_term_id, include="both", hops=0, wrap_width=20):
    """
    Plot a GO term subgraph using the provided GO DAG and GO term ID.

    Parameters:
    go_dag (networkx.DiGraph): GO DAG containing the ontology.
    go_term_id (str): GO term ID to plot.
    include (str): Whether to include 'children', 'parents', or 'both'. Default is 'both'.
    hops (int): Number of hops to plot. 0 represents the full plot with all ancestors and children.
    wrap_width (int): Maximum width of text before wrapping. Default is 20 characters.
    """

    # Load GO ontology
    G = go_dag

    # Helper functions
    def assign_edge_color(rel_type):
        color_map = {
            "is_a": "black",
            "part_of": "blue",
            "regulates": "green",
            "negatively_regulates": "red",
            "positively_regulates": "purple",
        }
        return color_map.get(rel_type, "grey")

    def adjust_figure_size(subgraph):
        num_nodes = len(subgraph.nodes())
        height = max(8, num_nodes * 0.5)
        width = height * 3
        return (width, height)

    def add_legend(namespace_color_map):
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D

        namespace_patches = [
            Patch(color=color, label=ns) for ns, color in namespace_color_map.items()
        ]

        edge_lines = [
            Line2D([0], [0], color="black", lw=2, label="is_a"),
            Line2D([0], [0], color="blue", lw=2, label="part_of"),
            Line2D([0], [0], color="green", lw=2, label="regulates"),
            Line2D([0], [0], color="red", lw=2, label="negatively_regulates"),
            Line2D([0], [0], color="purple", lw=2, label="positively_regulates"),
        ]

        plt.legend(
            handles=namespace_patches + edge_lines,
            loc="center left",
            bbox_to_anchor=(1.0, 0.5),
            frameon=False,
        )

    def get_nodes_within_hops(node, hops, direction):
        """Retrieve nodes within a specified number of hops."""
        nodes = {node}
        if direction == "descendants":
            for i in range(hops):
                next_level = set()
                for n in nodes:
                    next_level.update(nx.descendants_at_distance(G, n, i + 1))
                nodes.update(next_level)
        elif direction == "ancestors":
            for i in range(hops):
                next_level = set()
                for n in nodes:
                    next_level.update(nx.ancestors_at_distance(G, n, i + 1))
                nodes.update(next_level)
        return nodes

    # Check if the GO term exists in the ontology
    if go_term_id not in G:
        print(f"GO term {go_term_id} not found in the ontology.")
        return

    # Get subgraph (parents, children, or both) with hops
    if include == "children":
        if hops == 0:
            descendants = nx.descendants(G, go_term_id)
        else:
            descendants = get_nodes_within_hops(go_term_id, hops, "descendants")
        subgraph_nodes = {go_term_id}.union(descendants)
    elif include == "parents":
        if hops == 0:
            ancestors = nx.ancestors(G, go_term_id)
        else:
            ancestors = get_nodes_within_hops(go_term_id, hops, "ancestors")
        subgraph_nodes = {go_term_id}.union(ancestors)
    elif include == "both":
        if hops == 0:
            descendants = nx.descendants(G, go_term_id)
            ancestors = nx.ancestors(G, go_term_id)
        else:
            descendants = get_nodes_within_hops(go_term_id, hops, "descendants")
            ancestors = get_nodes_within_hops(go_term_id, hops, "ancestors")
        subgraph_nodes = {go_term_id}.union(descendants).union(ancestors)

    subgraph = G.subgraph(subgraph_nodes).copy()

    # Extract relationships and edge colors
    relationships = []
    edge_colors = {}

    for go_id in subgraph.nodes():
        attributes = G.nodes[go_id]

        if "is_a" in attributes:
            for parent_id in attributes["is_a"]:
                # Check if the edge exists in the GO DAG
                if G.has_edge(go_id, parent_id):
                    relationships.append((go_id, parent_id, "is_a"))
                    edge_colors[(go_id, parent_id)] = "black"

        if "relationship" in attributes:
            for item in attributes["relationship"]:
                rel_type, parent_id = item.split(" ", 1)
                # Check if the edge exists in the GO DAG
                if G.has_edge(go_id, parent_id):
                    relationships.append((go_id, parent_id, rel_type))
                    edge_colors[(go_id, parent_id)] = assign_edge_color(rel_type)

    # Plot GO subgraph
    fig_size = adjust_figure_size(subgraph)
    plt.figure(figsize=fig_size)

    A = nx.nx_agraph.to_agraph(subgraph)

    namespace_color_map = {
        "biological_process": "#7FDB7F",
        "molecular_function": "#0074D9",
        "cellular_component": "#FF851B",
    }

    query_color_map = {
        "biological_process": "#3D9970",
        "molecular_function": "#39CCCC",
        "cellular_component": "#FFDC00",
    }

    for node in A.nodes():
        node_name = subgraph.nodes[node].get("name", "Unknown")
        namespace = subgraph.nodes[node].get("namespace", "unknown")

        # Wrap the text for better readability
        wrapped_name = "\n".join(textwrap.wrap(node_name, width=wrap_width))
        A.get_node(node).attr["label"] = f"{wrapped_name}\n{node}"
        A.get_node(node).attr["shape"] = "rect"
        A.get_node(node).attr["style"] = "filled"

        if node == go_term_id:
            A.get_node(node).attr["fillcolor"] = query_color_map.get(
                namespace, "yellow"
            )
        else:
            A.get_node(node).attr["fillcolor"] = namespace_color_map.get(
                namespace, "lightgrey"
            )

    for rel in relationships:
        child, parent, rel_type = rel
        if child in subgraph and parent in subgraph:
            edge = A.get_edge(child, parent)
            edge.attr["color"] = edge_colors.get((child, parent), "grey")

    A.graph_attr.update(rankdir="BT")
    A.layout(prog="dot")
    A.draw("go_subgraph_colored.png")

    img = plt.imread("go_subgraph_colored.png")
    plt.imshow(img)
    plt.axis("off")

    add_legend(namespace_color_map)

    plt.show()


def generate_go_subgraphs(go_dag, output_dir):
    """
    Generate subgraphs from the GO ontology and save each subgraph as a Torch tensor and a NetworkX graph,
    starting from the leaf nodes upwards.

    Parameters:
    go_dag (NetworkX DiGraph): The GO ontology loaded as a NetworkX directed graph.
    output_dir (str): Directory where the Torch tensors and NetworkX graphs will be saved.
    """
    # Create directories for tensors and graphs if they do not exist
    tensor_output_dir = os.path.join(output_dir, "tensors")
    graph_output_dir = os.path.join(output_dir, "graphs")

    if not os.path.exists(tensor_output_dir):
        os.makedirs(tensor_output_dir)
    if not os.path.exists(graph_output_dir):
        os.makedirs(graph_output_dir)

    # Load GO ontology (assumed already loaded as go_dag)
    G = go_dag

    # Identify leaf nodes (nodes with no children)
    leaf_nodes = [n for n, d in G.out_degree() if d == 0]

    def convert_to_torch_tensor(subgraph):
        """Convert a NetworkX subgraph to PyTorch tensor format."""
        # Extract nodes and edges from the subgraph
        nodes = list(subgraph.nodes())
        node_idx = {node: i for i, node in enumerate(nodes)}
        edges = [(node_idx[u], node_idx[v]) for u, v in subgraph.edges()]

        # Create Torch tensor from node features (can be customized as needed)
        x = torch.eye(
            len(nodes), dtype=torch.float
        )  # Identity matrix as dummy node features

        # Create edge index tensor
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        # Create a Data object (PyTorch Geometric style)
        data = Data(x=x, edge_index=edge_index)

        return data

    def get_subgraph_and_save(go_term_id):
        """
        Get subgraph for the given GO term, save it as both a Torch tensor and a NetworkX graph.
        """
        ancestors = nx.ancestors(G, go_term_id)
        subgraph_nodes = {go_term_id}.union(ancestors)
        subgraph = G.subgraph(subgraph_nodes).copy()

        # Convert subgraph to tensor
        tensor_data = convert_to_torch_tensor(subgraph)

        # Save tensor to file
        torch.save(tensor_data, os.path.join(tensor_output_dir, f"{go_term_id}.pt"))

        # Save subgraph as NetworkX graph using pickle
        with open(os.path.join(graph_output_dir, f"{go_term_id}.gpickle"), "wb") as f:
            pickle.dump(subgraph, f)

    # Generate subgraphs and save both tensors and graphs starting from the leaf nodes
    visited = set()
    for leaf in tqdm(leaf_nodes, total=len(leaf_nodes), desc="Generating subgraphs..."):
        # Traverse each leaf node upwards and generate subgraphs
        ancestors = nx.ancestors(G, leaf)
        for ancestor in ancestors.union({leaf}):
            if ancestor not in visited:
                get_subgraph_and_save(ancestor)
                visited.add(ancestor)

    print(f"Saved subgraphs for {len(visited)} GO terms to {output_dir}")
