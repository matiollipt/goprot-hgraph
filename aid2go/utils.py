import re
import math
from pathlib import Path
import pandas as pd
import numpy as np
import requests
import ftplib
import bitmath
from Bio import SeqIO
from obonet import read_obo


def download_ftp(
    ftp_url="ftp.ebi.ac.uk",
    ftp_directory="/pub/databases/GO/goa/UNIPROT/",
    local_dir=None,
    filename=None,
    list_ftp_dir=False,
):
    """
    Downloads a file from the specified FTP server and saves it to the local directory.

    Parameters:
    ftp_url (str): The URL of the FTP server. Default is "ftp.ebi.ac.uk".
    ftp_directory (str): The directory on the FTP server where the file is located. Default is "/pub/databases/GO/goa/UNIPROT/".
    local_dir (str or Path): The local directory where the file will be saved. If None, the file will not be saved.
    filename (str): The name of the file to be downloaded.
    list_ftp_dir (bool): If True, the function will list the files available on the FTP server. Default is False.

    Returns:
    None
    """
    if local_dir is not None and filename is not None:

        # Ensures Path Object
        local_dir = Path(local_dir)

        # Local directory
        local_dir.mkdir(parents=True, exist_ok=True)

        # Define local filepath
        local_filepath = local_dir / filename

        # Connect to the FTP server
        ftp = ftplib.FTP(ftp_url)
        ftp.login()  # Anonymous login
        ftp.cwd(ftp_directory)

        # List directory contents
        if list_ftp_dir:
            print("Files available on the FTP server:")
            ftp.retrlines("LIST")

        # Get file size in readable format
        file_size_in_bytes = ftp.size(filename)
        file_size_in_mb = float(bitmath.Byte(file_size_in_bytes).to_MB())
        print(f"\nThe size of {filename} is {file_size_in_mb:.2f} MB")

        # Initialize variables for progress tracking
        downloaded_size = 0

        # Function to track download progress
        def download_with_progress(block):
            nonlocal downloaded_size  # Track the progress within this function
            downloaded_size += len(block)
            progress = (downloaded_size / file_size_in_bytes) * 100
            progress_mb = float(bitmath.Byte(downloaded_size).to_MB())
            print(
                f"\rDownload progress: {progress:.2f}% ({progress_mb:.2f} MB)", end=""
            )

        # Download the file
        with open(local_filepath, "wb") as file:
            ftp.retrbinary(
                f"RETR {filename}",
                lambda block: (file.write(block), download_with_progress(block)),
            )

        # Confirming the download
        print(f"\nDownloaded {filename} successfully to {local_filepath}")

        # Quit the FTP connection
        ftp.quit()


def load_fastas(filepath):
    """
    Loads protein sequences from a FASTA file.

    Parameters:
        filepath (str): The path to the FASTA file.

    Returns:
        sequences_dict (dict): A dictionary where keys are UniProt IDs and values are protein sequences.
    """
    sequences_dict = {}  # Dictionary to store sequences
    with open(filepath) as handle:
        for record in SeqIO.parse(handle, "fasta"):
            uniprot_id = record.id.split("|")[1]
            sequences_dict[uniprot_id] = str(record.seq)

    print(f"Loaded {len(sequences_dict)} fasta sequences.")
    return sequences_dict


def square_layout(num_of_plots: int) -> tuple[int, int]:
    """
    Calculate the optimal number of rows and columns for a square layout given the total number of plots.

    Parameters:
    num_of_plots (int): The total number of plots to be displayed.

    Returns:
    tuple[int, int]: A tuple containing the optimal number of rows and columns for the square layout.
    """
    ncols = math.ceil(math.sqrt(num_of_plots))
    nrows = math.ceil(num_of_plots / ncols)
    return nrows, ncols


def download_uniprot_idmapping(save_path):
    """
    Downloads the UniProtKB ID mapping file from the EBI FTP server.

    Returns:
    None
    """

    # FTP server details
    ftp_url = "ftp.ebi.ac.uk"
    ftp_directory = "/pub/databases/uniprot/current_release/knowledgebase/idmapping/"
    filename = "idmapping.dat.gz"

    download_ftp(ftp_url, ftp_directory, save_path, filename)
