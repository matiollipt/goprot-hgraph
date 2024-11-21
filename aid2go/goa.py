import pandas as pd
from tqdm import tqdm
from obonet import read_obo

# Column names GAF2.1 (Gene Association File)
gaf_columns = [
    "db",
    "uniprot_id",
    "symbol",
    "qualifier",
    "go_id",
    "reference",
    "evidence",
    "with_from",
    "aspect",
    "name",
    "synonym",
    "type",
    "taxon",
    "date",
    "assigned_by",
    "extension",
    "product_form_id",
]

# Mapping of GAF2.1 column names to shorter identifiers
gaf_mapping = {
    "DB": "db",
    "DB_Object_ID": "uniprot_id",
    "DB_Object_Symbol": "symbol",
    "Qualifier": "qualifier",
    "GO_ID": "go_id",
    "DB:Reference": "reference",
    "Evidence Code": "evidence",
    "With (or) From": "with_from",
    "Aspect": "aspect",
    "DB_Object_Name": "name",
    "DB_Object_Synonym": "synonym",
    "DB_Object_Type": "type",
    "Taxon and Interacting taxon": "taxon",
    "Date": "date",
    "Assigned_By": "assigned_by",
    "Annotation_Extension": "extension",
    "Gene_Product_Form_ID": "product_form_id",
}

# Dictionaries with evidence codes for filtering GAF files
experimental_evidence_codes = {
    "EXP": "Inferred from Experiment",
    "IDA": "Inferred from Direct Assay",
    "IPI": "Inferred from Physical Interaction",
    "IMP": "Inferred from Mutant Phenotype",
    "IGI": "Inferred from Genetic Interaction",
    "IEP": "Inferred from Expression Pattern",
}

cafa_evidence_codes = {
    "EXP": "Inferred from Experiment",
    "IDA": "Inferred from Direct Assay",
    "IMP": "Inferred from Mutant Phenotype",
    "IGI": "Inferred from Genetic Interaction",
    "IEP": "Inferred from Expression Pattern",
    "IC": "Inferred by Curator",
    "TAS": "Traceable Author Statement",
}

highthroughput_evidence_codes = {
    "HTP": "Inferred from High Throughput Experiment",
    "HDA": "Inferred from High Throughput Direct Assay",
    "HMP": "Inferred from High Throughput Mutant Phenotype",
    "HGI": "Inferred from High Throughput Genetic Interaction",
    "HEP": "Inferred from High Throughput Expression Pattern",
}

statement_evidence_codes = {
    "IC": "Inferred by Curator",
    "TAS": "Traceable Author Statement",
}

electronic_evidence_codes = {
    "IEA": "Inferred from Electronic Annotation",
    "ISS": "Inferred from Sequence or Structural Similarity",
    "ISO": "Inferred from Sequence Orthology",
    "ISA": "Inferred from Sequence Alignment",
    "ISM": "Inferred from Sequence Model",
    "IGC": "Inferred from Genomic Context",
    "RCA": "Inferred from Reviewed Computational Analysis",
    "IBA": "Inferred from Biological aspect of Ancestor",
    "IBD": "Inferred from Biological aspect of Descendant",
    "IKR": "Inferred from Key Residues",
    "IRD": "Inferred from Rapid Divergence",
}

low_confidence_evidence_codes = {
    "NAS": "Non-traceable Author Statement",
    "ND": "No biological Data available",
}

# Combine all codes into a single dict for later data analysis
all_evidence_codes = {
    **experimental_evidence_codes,
    **statement_evidence_codes,
    **electronic_evidence_codes,
    **low_confidence_evidence_codes,
}

import pandas as pd
from tqdm import tqdm
from pathlib import Path
import obonet


def filter_goa_by_evidence(
    gaf_file,
    evidence_codes,
    chunk_size=1000000,
    max_chunks=None,
    remove_obsolete_terms=True,
    generate_annot_file=True,
    save_path=None,
):
    """
    Filters a GAF file by specified evidence codes.

    :param gaf_file: Path to the GAF file (gzipped)
    :param evidence_codes: List or set of evidence codes to filter by
    :param chunk_size: Number of rows to process per chunk
    :param max_chunks: Maximum number of chunks to process (None for no limit)
    :param remove_obsolete_terms: If True, obsolete GO terms will be removed from the output DataFrame
    :param generate_annot_file: If True, generates an annotation file for IC calculation
    :param save_path: Path to save the filtered DataFrame and annotation file (None for no save)
    :return: DataFrame of filtered annotations, DataFrame of filtered annotations for IC calculation ([EntryID, term, aspect] columns) and evidence code frequency
    """

    # Define the columns as per GAF format
    gaf_columns = [
        "DB",
        "DB_Object_ID",
        "DB_Object_Symbol",
        "Qualifier",
        "GO_ID",
        "DB_Reference",
        "evidence",
        "With_From",
        "Aspect",
        "DB_Object_Name",
        "DB_Object_Synonym",
        "DB_Object_Type",
        "Taxon",
        "Date",
        "Assigned_By",
        "Annotation_Extension",
        "Gene_Product_Form_ID",
    ]

    # Validate save_path as a Path object if provided
    if save_path is not None:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

    # Initialize data chunks
    evidence_code_freq = {}
    filtered_chunks = []
    n_chunks = 0

    # Process the GAF file in chunks
    try:
        for chunk in tqdm(
            pd.read_csv(
                gaf_file,
                sep="\t",
                comment="!",
                names=gaf_columns,
                header=None,
                compression="gzip",
                chunksize=chunk_size,
                low_memory=False,
            ),
            total=1192255358 // chunk_size,
            desc="Loading and filtering GAF file in chunks",
        ):
            # Get evidence code statistics
            evidence_codes_in_chunk = chunk["evidence"].value_counts().to_dict()
            for code, count in evidence_codes_in_chunk.items():
                evidence_code_freq[code] = evidence_code_freq.get(code, 0) + count

            # Filter by the provided evidence codes
            filtered_chunk = chunk[chunk["evidence"].isin(evidence_codes)]
            filtered_chunks.append(filtered_chunk)

            # Counter
            n_chunks += 1
            if max_chunks is not None and n_chunks >= max_chunks:
                break
    except FileNotFoundError:
        print(f"Error: The file {gaf_file} was not found.")
        return None, None
    except pd.errors.ParserError:
        print("Error: There was an issue parsing the GAF file.")
        return None, None

    # Concatenate all filtered chunks into a single DataFrame
    filtered_df = pd.concat(filtered_chunks, ignore_index=True)

    if remove_obsolete_terms:
        # Get GO DAG with valid terms
        go_dag = obonet.read_obo(
            "http://purl.obolibrary.org/obo/go/go-basic.obo", ignore_obsolete=True
        )
        valid_go_ids = set(go_dag.nodes())
        filtered_df = filtered_df[filtered_df["GO_ID"].isin(valid_go_ids)]
        filtered_df.reset_index(drop=True, inplace=True)

    # Rename columns
    filtered_df.rename(columns=gaf_mapping, inplace=True)

    annot_df = None
    if generate_annot_file:
        # Create annotation file to calculate IC
        annot_df = filtered_df[["uniprot_id", "go_id", "aspect"]].copy()
        annot_df.rename(
            columns={"uniprot_id": "EntryID", "go_id": "term"}, inplace=True
        )
        annot_df["aspect"] = annot_df["aspect"].map(
            {"F": "MFO", "C": "CCO", "P": "BPO"}
        )

    # Save the filtered DataFrame and annotation file if required
    if save_path is not None:
        # Save the filtered DataFrame
        filtered_df_filepath = save_path / "goa_hc.tsv"
        filtered_df.to_csv(filtered_df_filepath, sep="\t", index=False)

        if generate_annot_file and annot_df is not None:
            # Save annotation file
            annot_df_filepath = save_path / "goa_hc_annot.tsv"
            annot_df.to_csv(annot_df_filepath, index=False, sep="\t")

        # Save the evidence counts (all GOA entries)
        evidence_counts_filepath = save_path / "evidence_counts_all.tsv"
        with open(evidence_counts_filepath, "w") as f:
            for code, count in evidence_code_freq.items():
                f.write(f"{code}\t{count}\n")

    return filtered_df, annot_df, evidence_code_freq
