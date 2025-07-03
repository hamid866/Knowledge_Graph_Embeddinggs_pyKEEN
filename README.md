# Knowledge_Graph_Embeddinggs_pyKEEN
Knowledge Graph Embeddings with PyKEEN for HP and GO Ontologies
This project demonstrates large-scale knowledge graph embedding using PyKEEN on two biomedical ontologies:

Gene Ontology (GO): A widely-used ontology describing gene functions.

Human Phenotype Ontology (HP): An ontology capturing human phenotypic abnormalities.

Pipeline Overview
Ontology Loading

Both HP and GO ontologies are loaded (from local .owl files or downloaded URLs) using rdflib.

Sample triples are extracted and displayed for verification.

Triple Extraction and Data Preparation

Entities (nodes) and relationships (edges) are parsed and saved in both CSV and JSON formats.

A unified knowledge graph is constructed from both ontologies and exported as a triples file (kg_triples.txt) for downstream embedding.

Embedding Model Training

Five different models are trained using PyKEEN:

Dismult

TransE

TransR

BoxE

HolE

For each model:

The knowledge graph triples are split into train/validation/test sets.

The chosen model is trained for a specified number of epochs.

Embeddings for all entities and relations are extracted and saved as JSON dictionaries (easy to use for downstream tasks).

Output

For each model, two files are generated:

<model_name>_entity.json: Dictionary mapping entity URIs to embedding vectors.

<model_name>_relation.json: Dictionary mapping relation types to embedding vectors.

These JSON files can be used directly for similarity calculation, downstream ML models, or as a service in web APIs.

How to Use
Make sure all dependencies are installed with pip install -r requirements.txt.

Place the ontology files (go-basic.owl and hp.owl) in the project directory (or update the paths in the script).

Run the script to train the models:

bash
Copy
Edit
python pykeen_embedding_script.py
After training, embeddings for each model will be available in the embeddings directory.

Dependencies
Python 3.8+

PyKEEN

PyTorch

pandas

rdflib

All dependencies are listed in requirements.txt.

Notes
You can adjust the number of epochs, embedding dimensions, and model types in the script as needed.

The project is suitable for anyone working on biomedical knowledge graphs, entity similarity, tasks using HP and GO ontologies.
