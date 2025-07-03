#!/usr/bin/env python
# coding: utf-8

import os
import pickle
import csv
import json
import pandas as pd
import torch
from rdflib import Graph, Namespace, RDF, RDFS
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory

######################################
# Ontology Loading and Extraction
######################################

# URLs and Local Paths
gene_ontology_url = "https://purl.obolibrary.org/obo/go/go-basic.owl"
hoo_ontology_path = "hp.owl"  # Update this path if needed

def load_ontology(source, is_local=False):
    """Loads an ontology from a URL or local file."""
    g = Graph()
    if is_local:
        print(f"Loading ontology from local file: {source}...")
        g.parse(source, format="xml")
    else:
        print(f"Downloading and loading ontology from: {source}...")
        g.parse(source, format="xml")
    print(f"Loaded {len(g)} triples.")
    return g

# Load ontologies
go_graph = load_ontology(gene_ontology_url, is_local=False)
if os.path.exists(hpo_ontology_path):
    hop_graph = load_ontology(hpo_ontology_path, is_local=True)
else:
    print(f"Error: The file {hpo_ontology_path} was not found. Please check the path.")

def show_sample_triples(graph, name, num=5):
    """Prints sample triples from the graph."""
    print(f"\nSample triples from {name}:")
    for s, p, o in list(graph)[:num]:
        print(s, p, o)

show_sample_triples(go_graph, "Gene Ontology")
if 'hpo_graph' in locals():
    show_sample_triples(hpo_graph, "Human Phenotype Ontology")

######################################
# Data Extraction to CSV/JSON Files
######################################

# File Paths
entity_csv = "entities.csv"
relationship_csv = "relationships.csv"
definition_csv = "definitions.csv"

entity_json = "entities.json"
relationship_json = "relationships.json"
definition_json = "definitions.json"

def extract_entities(graph):
    """Extracts entity labels and saves them."""
    entities = []
    for s, p, o in graph.triples((None, RDFS.label, None)):
        entities.append({"id": str(s), "label": str(o)})
    with open(entity_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "label"])
        writer.writeheader()
        writer.writerows(entities)
    with open(entity_json, "w") as f:
        json.dump(entities, f, indent=4)
    print(f"✅ Saved {len(entities)} entities.")

def extract_relationships(graph):
    """Extracts relationships (edges) and saves them."""
    relationships = []
    for s, p, o in graph.triples((None, RDFS.subClassOf, None)):
        relationships.append({"source": str(s), "relation": str(p), "target": str(o)})
    with open(relationship_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["source", "relation", "target"])
        writer.writeheader()
        writer.writerows(relationships)
    with open(relationship_json, "w") as f:
        json.dump(relationships, f, indent=4)
    print(f"✅ Saved {len(relationships)} relationships.")

def extract_definitions(graph):
    """Extracts term definitions and saves them."""
    OBO_DEFINITION = Namespace("http://purl.obolibrary.org/obo/IAO_0000115")
    definitions = []
    for s, p, o in graph.triples((None, OBO_DEFINITION, None)):
        definitions.append({"id": str(s), "definition": str(o)})
    with open(definition_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "definition"])
        writer.writeheader()
        writer.writerows(definitions)
    with open(definition_json, "w") as f:
        json.dump(definitions, f, indent=4)
    print(f"✅ Saved {len(definitions)} definitions.")

# Run the extraction on both ontologies
extract_entities(go_graph)
if 'hpo_graph' in locals():
    extract_entities(hpo_graph)

extract_relationships(go_graph)
if 'hpo_graph' in locals():
    extract_relationships(hpo_graph)

extract_definitions(go_graph)
if 'hpo_graph' in locals():
    extract_definitions(hpo_graph)

print("✅ All data saved successfully!")

######################################
# Prepare Data for Embedding Models
######################################

# Use the relationships CSV to create a Knowledge Graph triples file
df = pd.read_csv("relationships.csv")
kg_data = df[["source", "relation", "target"]]
kg_data.to_csv("kg_triples.txt", sep="\t", index=False, header=False)
print("✅ Knowledge Graph data saved for embedding models!")

######################################
# Train Embedding Models with PyKEEN
# and Save Embeddings as Dictionaries
######################################

# Read KG triples and create TriplesFactory
kg_path = "kg_triples.txt"
df_triples = pd.read_csv(kg_path, sep="\t", names=["head", "relation", "tail"])
triples = df_triples[["head", "relation", "tail"]].values
dataset = TriplesFactory.from_labeled_triples(triples)
train_factory, valid_factory, test_factory = dataset.split([0.8, 0.1, 0.1], random_state=42)

# Create directory to save embeddings
embedding_dir = "embeddings"
os.makedirs(embedding_dir, exist_ok=True)

models_to_train = [‘put All models name’]
results = {}

for model_name in models_to_train:
    print(f"\nTraining {model_name} with embedding_dim=100 ...")
    result = pipeline(
        model=model_name,
        training=train_factory,
        validation=valid_factory,
        testing=test_factory,
        training_loop="lcwa",
        epochs=100,
        model_kwargs={"embedding_dim": 200},
        use_tqdm=True
    )
    
    results[model_name] = result
    model = result.model

    # Get PyKEEN embedding objects
    entity_embeddings = model.entity_representations[0]
    relation_embeddings = model.relation_representations[0]
    
    # Retrieve mapping from entity/relation to internal IDs
    # Option 1: Use your original TriplesFactory from the dataset
    entity_to_id = dataset.entity_to_id
    relation_to_id = dataset.relation_to_id


    # Convert entity embeddings to a dictionary
    entity_dict = {}
    for entity, idx in entity_to_id.items():
        embedding_tensor = entity_embeddings(torch.tensor([idx]))
        entity_dict[entity] = embedding_tensor.detach().cpu().numpy().tolist()[0]
  

    # Convert relation embeddings to a dictionary
    relation_dict = {}
    for relation, idx in relation_to_id.items():
        embedding_tensor = relation_embeddings(torch.tensor([idx]))
        relation_dict[relation] = embedding_tensor.detach().cpu().numpy().tolist()[0]



    # ALSO Save the dictionaries to JSON
    with open(os.path.join(embedding_dir, f"{model_name}_entity.json"), "w") as f:
        json.dump(entity_dict, f, indent=4)
    with open(os.path.join(embedding_dir, f"{model_name}_relation.json"), "w") as f:
        json.dump(relation_dict, f, indent=4)

    print(f"✅ Saved {model_name} embeddings as both pickle and JSON dictionaries!")

print("✅ All models trained and embeddings saved successfully!")

######################################
# Load and Verify Saved Embeddings (Optional)
######################################

def load_embeddings_json(model_name):

    with open(os.path.join(embedding_dir, f"{model_name}_entity.json"), "r") as f:
        entity_embeddings = json.load(f)
    with open(os.path.join(embedding_dir, f"{model_name}_relation.json"), "r") as f:
        relation_embeddings = json.load(f)
    print(f"✅ Loaded {model_name} JSON embeddings successfully.")
    return entity_embeddings, relation_embeddings


# Example: Load and print a sample of the TransE embeddings
transE_entity_emb, transE_relation_emb = load_embeddings("TransE")
print("Sample entity embedding:")
print(list(transE_entity_emb.items())[0])
