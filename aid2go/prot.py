from typing import List, Tuple, Union
import time
import sys
import os
import re
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
from Bio import SeqIO
import networkx as nx
import requests

import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.utils import degree, to_undirected
from torch_geometric.data import Data
from transformers import BertTokenizer, BertModel

# UniPROT REST API functions
UNIPROT_API = "https://rest.uniprot.org"  # UniProt REST API endpoint


def get_url(url, **kwargs):
    response = requests.get(url, **kwargs)
    if not response.ok:
        print(response.text)
        response.raise_for_status()
        sys.exit()
    return response


def check_response(response):
    try:
        response.raise_for_status()
    except:
        print(response.json())
        raise


def submit_id_mapping(from_db, to_db, ids):
    """
    Submits a job to the UniProt REST API for mapping protein identifiers from one database to another.

    Parameters:
    from_db (str): The source database from which the protein identifiers are to be mapped.
    to_db (str): The target database to which the protein identifiers are to be mapped.
    ids (list): A list of protein identifiers to be mapped.

    Returns:
    job_id (str): The job ID returned by the UniProt API for tracking the mapping job.

    Raises:
    requests.exceptions.RequestException: If there is an error with the HTTP request.
    """
    request = requests.post(
        f"{UNIPROT_API}/idmapping/run",
        data={"from": from_db, "to": to_db, "ids": ",".join(ids)},
    )
    check_response(request)
    return request.json()["jobId"]


def fetch_all_results(job_id, api_url):
    """
    Fetches all paginated results from a UniProt ID mapping job.

    Parameters:
    - job_id: The job ID returned by the submit_id_mapping function.
    - api_url: The base URL for the UniProt API.

    Returns:
    - A list of all mapped results.
    """
    results = []
    next_url = f"{api_url}/idmapping/results/{job_id}"

    # Fetch the first page of results
    r = requests.get(next_url)

    # Get the total number of results (for reporting or debugging)
    total = r.headers.get("x-total-results")
    data = r.json()

    # Store the results from the first page
    results.extend(data.get("results", []))

    # Print progress
    print(f"Total results: {total}; Fetched {len(results)} so far...")

    # Keep fetching next pages if available
    while r.links.get("next", {}).get("url"):
        next_url = r.links["next"]["url"]
        r = requests.get(next_url)
        data = r.json()

        # Append the next page of results
        results.extend(data.get("results", []))

        # Report progress
        print(f"Fetched {len(results)} results so far...")

    return results


class ProteinSequenceDataset(Dataset):
    """
    A PyTorch Dataset class for processing protein sequences from a dictionary.

    Attributes:
        sequences_dict (dict): A dictionary where keys are UniProt IDs and values are sequences.
        max_length (int): The maximum allowed sequence length.
        min_length (int): The minimum allowed sequence length.

    Methods:
        __len__: Returns the number of sequences in the dataset.
        __getitem__: Retrieves a sequence and its identifier by index, applying preprocessing.
    """

    def __init__(self, sequences_dict, max_length, min_length=50):
        """
        Initializes the ProteinSequenceDataset with a dictionary of sequences and a sequence length range.

        Parameters:
            sequences_dict (dict): The dictionary of UniProt IDs and sequences.
            max_length (int): The maximum allowed sequence length.
            min_length (int): The minimum allowed sequence length.
        """
        self.sequences_dict = sequences_dict
        self.max_length = max_length
        self.min_length = min_length
        self.ids = list(sequences_dict.keys())

    def __len__(self):
        """
        Provides the size of the dataset.

        Returns:
            int: The number of sequences in the dataset.
        """
        return len(self.ids)

    def __getitem__(self, idx):
        """
        Retrieves the sequence and its identifier at the specified index after preprocessing.

        Parameters:
            idx (int): The index of the sequence to retrieve.

        Returns:
            tuple: A tuple containing the preprocessed sequence and its UniProt ID.
        """
        seq_id = self.ids[idx]
        seq = self.sequences_dict[seq_id]
        seq = re.sub(r"[UZOB]", "X", seq.replace("\n", "").replace("", " "))
        return seq, seq_id


def embed_sequences_from_dict(
    sequences_dict,
    per_residue=False,
    batch_size=32,
    max_length=4000,
    min_length=50,
    save_path="embeddings",
):
    """
    Processes protein sequences from a dictionary, generating embeddings using a pre-trained BERT model.

    Parameters:
        sequences_dict (dict): Dictionary where keys are UniProt IDs and values are sequences.
        per_residue (bool): If True, saves per-residue embeddings; otherwise, saves per-protein embeddings.
        batch_size (int): The batch size for processing sequences.
        max_length (int): Sequences longer than this value will be skipped.
        min_length (int): Sequences shorter than this value will be skipped.
        save_path (str): Path where the embeddings will be saved.

    Returns:
        dict: A dictionary of embeddings with UniProt ID as key and embedding tensor as value.
        dict: A dictionary of parsed sequences with UniProt ID as key and parsed sequence as value.
        dict: A dictionary of large proteins (not embedded) with UniProt ID as key and sequence as value.
        dict: A dictionary of small proteins (not embedded) with UniProt ID as key and sequence as value.
    """
    # Create directory to save the embeddings
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading transformer and tokenizer...")
    tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
    model = BertModel.from_pretrained("Rostlab/prot_bert")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).half().eval()  # Move model to GPU and set half-precision (float16)

    # Creates directories to save embeddings
    embeddings_folder = Path(save_path)
    embeddings_folder.mkdir(parents=True, exist_ok=True)
    per_protein_path = embeddings_folder / "per_protein"
    per_protein_path.mkdir(exist_ok=True)
    per_residue_path = embeddings_folder / "per_residue" if per_residue else None
    if per_residue_path:
        per_residue_path.mkdir(exist_ok=True)

    # Inspect saved embeddings and find already processed proteins
    existing_embeds = set([file.stem for file in per_protein_path.glob("*.pt")])

    if existing_embeds:
        print(f"Skipping {len(existing_embeds)} existing embeddings.")

    # Dataset and DataLoader
    dataset = ProteinSequenceDataset(sequences_dict, max_length, min_length)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=False)

    # Dictionary to keep large proteins (not embedded)
    large_prots = {}
    # Dictionary to keep small proteins (not embedded)
    small_prots = {}
    # Dictionary to keep embeddings
    embeddings_dict = {}
    # Dictionary to store parsed sequences
    parsed_sequences = {}

    for seqs, ids in tqdm(loader, desc="Embedding sequences"):
        for seq, seq_id in zip(seqs, ids):

            # Skip sequences shorter than min_length
            if len(seq) < min_length:
                small_prots[seq_id] = seq
                continue

            # Check if embedding already exists
            if seq_id in existing_embeds:
                print(f"Skipping {seq_id}, already embedded.")
                continue

            # Check length after extracting the sequence
            if len(seq) >= max_length:
                large_prots[seq_id] = seq
                continue

            # Store the parsed sequence
            parsed_sequences[seq_id] = seq

            # Tokenize and move input tensors to the same device as the model
            inputs = tokenizer(seq, padding=True, truncation=True, return_tensors="pt")
            inputs = {key: tensor.to(device) for key, tensor in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)
            embeddings = outputs.last_hidden_state[:, : len(seq.replace(" ", ""))]

            # Detach and move to CPU before saving
            embeddings = embeddings.mean(dim=1).detach().cpu()
            torch.save(embeddings, per_protein_path / f"{seq_id}.pt")

            # Store the embedding in the dictionary
            embeddings_dict[seq_id] = embeddings

            if per_residue:
                torch.save(
                    embeddings.detach().cpu(),
                    per_residue_path / f"{seq_id}_res_embed.pt",
                )

    if large_prots:
        large_prots_df = pd.DataFrame(
            [{"uniprot_id": k, "sequence": v} for k, v in large_prots.items()]
        )
        large_prots_df.to_csv(embeddings_folder / "large_proteins.csv", index=False)
        print(f"Number of large proteins (not embedded): {len(large_prots_df)}")

    if small_prots:
        small_prots_df = pd.DataFrame(
            [{"uniprot_id": k, "sequence": v} for k, v in small_prots.items()]
        )
        small_prots_df.to_csv(embeddings_folder / "small_proteins.csv", index=False)
        print(f"Number of small proteins (not embedded): {len(small_prots_df)}")

    return embeddings_dict, parsed_sequences, large_prots, small_prots


class ProteinEmbeddingDataset(Dataset):
    """
    A PyTorch Dataset for loading protein embeddings stored as .pt (PyTorch tensor) files.

    Each tensor file contains the embedding of a protein, identified by a UniProt ID. This dataset
    facilitates the access to these embeddings for model training or inference, ensuring each
    embedding is loaded as needed, thus optimizing memory usage.

    Attributes:
        tensor_folder (str): The directory path where the embedding tensors (.pt files) are stored.
        uniprot_ids (list): A list of UniProt IDs corresponding to the proteins for which embeddings are available.

    Methods:
        __len__: Returns the total number of embeddings available in the dataset.
        __getitem__: Retrieves an embedding by its index, along with its corresponding UniProt ID.

    Parameters:
        tensor_folder (str): Path to the folder where tensor files, representing protein embeddings, are stored.
                             Each file should be named with the UniProt ID of the protein it represents.
    """

    def __init__(self, tensor_folder):
        """
        Initializes the dataset with the path to the directory containing the embeddings.

        Args:
            tensor_folder (string): Path to the folder where tensors are stored.
        """
        self.tensor_folder = tensor_folder
        self.uniprot_ids = [
            file.split(".")[0]
            for file in os.listdir(tensor_folder)
            if file.endswith(".pt")
        ]

    def __len__(self):
        """
        Returns the total number of protein embeddings available.

        Returns:
            int: The count of UniProt ID-based tensor files in the tensor folder.
        """
        return len(self.uniprot_ids)

    def __getitem__(self, idx):
        """
        Retrieves a specific protein embedding and its corresponding UniProt ID by index.

        This method ensures that the embedding tensor is loaded from disk only when accessed,
        optimizing memory usage, especially useful when dealing with large datasets.

        Args:
            idx (int): The index of the protein embedding to retrieve.

        Returns:
            dict: A dictionary containing the UniProt ID ('uniprot_id') and the embedding tensor ('tensor').
                  The tensor is expected to have a shape of (1, 1024).

        Raises:
            AssertionError: If the loaded tensor does not match the expected shape of (1, 1024).
        """
        uniprot_id = self.uniprot_ids[idx]
        tensor_path = os.path.join(self.tensor_folder, uniprot_id + ".pt")
        tensor = torch.load(tensor_path)

        # Asserting the tensor is in the correct shape ensures consistency and prevents potential errors in downstream processes.
        assert tensor.shape == (
            1,
            1024,
        ), f"Tensor shape mismatch: {tensor.shape} != (1, 1024)"

        return {"uniprot_id": uniprot_id, "tensor": tensor}


def PPIDataset(
    source_embeddings_dir,
    dest_embeddings_dir,
    associations_df,
    ppi_edges,
    ppi_nodes,
    dataset="train",
):
    """
    Process protein embeddings, create a Protein-Protein Interaction (PPI) graph, and save the data.

    This function loads protein embeddings from .pt files, filters them based on the provided associations,
    constructs a PPI graph, computes degrees, plots the degree distribution, and saves the processed data.

    Args:
        data_folder (Path): Path to the main data directory.
        ppi_folder (Path): Path to the PPI-specific folder where outputs will be saved.
        associations_df (pd.DataFrame): DataFrame containing protein associations with at least a 'uniprot_id' column.
        ppi_edges (list): List of tuples representing edges in the PPI network.
        ppi_nodes (list): List of nodes in the PPI network.
        dataset (str, optional): Identifier for the dataset (e.g., 'train', 'test'). Defaults to "train".

    Returns:
        torch_geometric.data.Data: The processed PPI data containing node features and edge indices.
    """

    # Load embeddings directly from the folder
    source_embeddings_dir = Path(source_embeddings_dir)  # Ensures Path Object
    source_embedding_files = [
        f for f in os.listdir(source_embeddings_dir) if f.endswith(".pt")
    ]
    print(
        f"# embeddings in {source_embeddings_dir.as_posix()}: {len(source_embedding_files)}"
    )

    # Collect embeddings
    unique_proteins = set(associations_df["uniprot_id"].unique())
    protein_embed_dict = {}
    for file in tqdm(source_embedding_files, desc="Loading Protein Embeddings"):
        uniprot_id = file.split(".")[0]
        if uniprot_id in unique_proteins:
            tensor_path = source_embeddings_dir / file
            tensor = torch.load(tensor_path)
            assert tensor.shape == (
                1,
                1024,
            ), f"Tensor shape mismatch: {tensor.shape} != (1, 1024)"
            protein_embed_dict[uniprot_id] = tensor.numpy().flatten()

    # Save embeddings
    with open(dest_embeddings_dir / "protein_embeddings.pkl", "wb") as file:
        pickle.dump(protein_embed_dict, file)

    # Process embeddings into a feature matrix
    protein_ids = list(protein_embed_dict.keys())
    protein_feats = torch.stack(
        [
            torch.tensor(protein_embed_dict[pid], dtype=torch.float32)
            for pid in protein_ids
        ]
    )
    protein_ids_map = {pid: idx for idx, pid in enumerate(protein_ids)}

    print(f"# of protein IDs: {len(protein_ids)}")
    print(f"Protein features shape: {protein_feats.shape}")

    print(protein_ids_map)

    # Create edge index
    edges_to_tensor = []
    missing_proteins = set()
    for e in ppi_edges:
        if e[0] in protein_ids_map and e[1] in protein_ids_map:
            edges_to_tensor.append((protein_ids_map[e[0]], protein_ids_map[e[1]]))
        else:
            missing_proteins.update([e[0], e[1]])

    if missing_proteins:
        print(f"{len(missing_proteins)} missing proteins: {missing_proteins}")

    if edges_to_tensor:
        edge_index_ppi = (
            torch.tensor(edges_to_tensor, dtype=torch.long).t().contiguous()
        )
        print(f"Edge index shape: {edge_index_ppi.shape}")
    else:
        print("No valid edges found!")

    # Convert to undirected graph and compute degrees
    edge_index_ppi = to_undirected(edge_index_ppi)
    degrees = degree(edge_index_ppi[0])

    # Plot degree distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(degrees.numpy(), bins=50, kde=True)
    plt.title("Distribution of Degrees")
    plt.xlabel("Degree")
    plt.ylabel("Frequency")
    plt.savefig(dest_embeddings_dir / f"degree_distribution_{dataset}.png")
    plt.close()

    # Create and save Data object
    ppi_data = Data(x=protein_feats, edge_index=edge_index_ppi)
    torch.save(ppi_data, dest_embeddings_dir / f"ppi_dataset_{dataset}.pt")

    return ppi_data, missing_proteins


def get_lcc(
    ppi_pairs: List[Tuple[str, str]],
    degree_threshold: int,
    graph_type: str = "undirected",
) -> Union[nx.Graph, nx.DiGraph]:
    """
    Create a graph (directed or undirected) from PPI pairs, filter nodes by degree (keeping nodes with degree <= degree_threshold),
    and return the largest connected component (LCC) with key statistics.

    Parameters:
    ----------
    ppi_pairs : List[Tuple[str, str]]
        List of tuples representing PPI (protein-protein interaction) pairs that form the graph edges.

    degree_threshold : int
        Maximum degree of nodes to keep. Only nodes with degree <= degree_threshold will be retained.

    graph_type : str, optional
        Type of graph to create: either 'directed' or 'undirected' (default is 'undirected').

    Returns:
    -------
    Union[nx.Graph, nx.DiGraph]
        The largest connected component (LCC) of the filtered graph (either directed or undirected).

    Raises:
    -------
    ValueError:
        If the provided graph_type is invalid (i.e., not 'directed' or 'undirected').

    Example:
    --------
    >>> ppi_pairs = [('P1', 'P2'), ('P2', 'P3'), ('P3', 'P4')]
    >>> get_lcc(ppi_pairs, degree_threshold=2, graph_type='undirected')
    """
    # Validate graph type
    if graph_type not in {"directed", "undirected"}:
        raise ValueError("graph_type must be either 'directed' or 'undirected'")

    # Create the graph based on the specified graph type
    if graph_type == "directed":
        ppi_graph = nx.DiGraph(ppi_pairs)  # Directed graph
    else:
        ppi_graph = nx.Graph(ppi_pairs)  # Undirected graph (default)

    # Filter nodes based on the degree threshold (keeping nodes with degree <= threshold)
    filtered_nodes = [
        node for node, degree in ppi_graph.degree() if degree <= degree_threshold
    ]

    # Create a subgraph with the filtered nodes
    ppi_graph_filtered = ppi_graph.subgraph(filtered_nodes).copy()

    # Get the largest connected component (LCC)
    if graph_type == "directed":
        # Use weakly connected components for directed graphs
        lcc_filtered = max(nx.weakly_connected_components(ppi_graph_filtered), key=len)
    else:
        # Use connected components for undirected graphs
        lcc_filtered = max(nx.connected_components(ppi_graph_filtered), key=len)

    ppi_graph_lcc = ppi_graph_filtered.subgraph(lcc_filtered).copy()

    # Print stats for the filtered graph and LCC
    _print_graph_stats(ppi_graph_filtered, ppi_graph_lcc, degree_threshold, graph_type)

    return ppi_graph_lcc


def _print_graph_stats(
    ppi_graph_filtered: Union[nx.Graph, nx.DiGraph],
    ppi_graph_lcc: Union[nx.Graph, nx.DiGraph],
    degree_threshold: int,
    graph_type: str,
) -> None:
    """
    Helper function to print graph statistics for the filtered graph and LCC.

    Parameters:
    ----------
    ppi_graph_filtered : Union[nx.Graph, nx.DiGraph]
        The filtered graph after degree threshold filtering.

    ppi_graph_lcc : Union[nx.Graph, nx.DiGraph]
        The largest connected component (LCC) of the filtered graph.

    degree_threshold : int
        The degree threshold used for filtering.

    graph_type : str
        Type of the graph ('directed' or 'undirected').
    """
    # Calculate average degree
    avr_degree = (
        sum(dict(ppi_graph_lcc.degree()).values()) / ppi_graph_lcc.number_of_nodes()
    )

    # Print-out basic graph attributes after filtering by degree
    print("\nFiltered PPI graph (by degree) attributes:")
    print(
        f"  - # of nodes (after filtering by degree <= {degree_threshold}): {ppi_graph_filtered.number_of_nodes()}"
    )
    print(f"  - # of edges (after filtering): {ppi_graph_filtered.number_of_edges()}")

    print("\nLargest Connected Component (LCC) attributes:")
    print(f"  - # of nodes (LCC): {ppi_graph_lcc.number_of_nodes()}")
    print(f"  - # of edges (LCC): {ppi_graph_lcc.number_of_edges()}")

    # Check if the LCC is connected
    is_connected = (
        nx.is_connected(ppi_graph_lcc)
        if graph_type == "undirected"
        else nx.is_weakly_connected(ppi_graph_lcc)
    )
    print(f"  - Is connected? (LCC): {is_connected}")
    print(f"  - Average degree (LCC): {avr_degree:.2f}")


import re
import requests
from requests.adapters import HTTPAdapter, Retry

# Regular expression to capture the 'next' link from the headers
re_next_link = re.compile(r'<(.+)>; rel="next"')

# Setup retries and session
retries = Retry(total=5, backoff_factor=0.25, status_forcelist=[500, 502, 503, 504])
session = requests.Session()
session.mount("https://", HTTPAdapter(max_retries=retries))


def get_next_link(headers):
    """Extract the 'next' pagination link from headers if available."""
    if "Link" in headers:
        match = re_next_link.match(headers["Link"])
        if match:
            return match.group(1)
    return None


def get_batch(batch_url):
    """Generator to handle paginated requests to the UniProt API."""
    while batch_url:
        response = session.get(batch_url)
        response.raise_for_status()  # Raise an error for bad responses
        total = response.headers.get("x-total-results", "unknown")
        yield response, total
        batch_url = get_next_link(response.headers)


def fetch_fasta_sequences_from_ids(uniprot_ids, batch_size=500):
    """
    Fetches protein sequences in FASTA format for a given list of UniProt IDs.

    Parameters:
    -----------
    uniprot_ids : list
        A list of UniProt IDs to fetch sequences from.
    batch_size : int
        The number of records to fetch per batch (default is 500).

    Returns:
    --------
    dict:
        A dictionary with UniProt IDs as keys and their corresponding sequences as values.
    """
    sequence_dict = {}

    # Create the query by joining UniProt IDs with " OR "
    query = " OR ".join([f"accession:{uniprot_id}" for uniprot_id in uniprot_ids])

    # Construct the initial URL for UniProt API with the query and batch size
    url = f"https://rest.uniprot.org/uniprotkb/search?fields=accession,sequence&format=fasta&query={query}&size={batch_size}"

    # Fetch sequences in batches, handling pagination
    for batch, total in get_batch(url):
        # Split the FASTA response into individual protein entries
        fasta_data = batch.text.split("\n>")

        for entry in fasta_data:
            # Handle the case where the first entry might not start with '>'
            if not entry.startswith(">"):
                entry = ">" + entry

            # Split the header and the sequence
            lines = entry.splitlines()
            header = lines[0]
            sequence = "".join(lines[1:])

            # Extract the UniProt ID from the header
            uniprot_id = header.split("|")[1]

            # Store the sequence in the dictionary
            sequence_dict[uniprot_id] = sequence

        print(f"{len(sequence_dict)} / {total} sequences fetched so far...")

    return sequence_dict
