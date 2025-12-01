#!/usr/bin/env python3
"""
Debug why propose_generator only processes 5 entities
"""
import json
from collections import defaultdict
from src.components.entity_graph_constructor import EntityRelationshipGraph

# Load the entity eliminator output
with open('src/outputs/entity_eliminator.json', 'r') as f:
    inputs = json.load(f)

# Create entity relationship graph
entity_relationship_graph = EntityRelationshipGraph(inputs)

print(f"Total items in entity_relationship_graph: {len(entity_relationship_graph.items())}")
print(f"Entity IDs in graph: {list(entity_relationship_graph.keys())[:20]}...")

# Simulate the propose_generator logic
already_done_entity_ids = {15, 18, 24, 27, 48}
skipped_no_relationships = 0
skipped_already_done = 0
eligible_count = 0

eligible_entities = []

for cur_entity_id, cur_entity_item in list(entity_relationship_graph.items())[:300]:
    if cur_entity_id in already_done_entity_ids:
        skipped_already_done += 1
        continue

    # Get subgraph
    subgraph_depth_1 = entity_relationship_graph.get_subgraph(cur_entity_id, depth=1)

    # Check relationships
    already_have_chunks = set()
    objective_relationships = []
    for relationship_item in subgraph_depth_1['relationships']:
        chunk_id = relationship_item['id']
        if chunk_id in already_have_chunks:
            continue
        already_have_chunks.add(chunk_id)
        objective_relationships.append(relationship_item)

    relationship_count = len(objective_relationships)

    if relationship_count == 0 or relationship_count <= 1:
        skipped_no_relationships += 1
        print(f"  Entity {cur_entity_id} skipped: {relationship_count} unique relationships")
        continue

    eligible_count += 1
    eligible_entities.append((cur_entity_id, relationship_count))

print(f"\nProcessing summary:")
print(f"  Already done: {skipped_already_done}")
print(f"  Skipped (≤1 relationships): {skipped_no_relationships}")
print(f"  Eligible for processing: {eligible_count}")

print(f"\nFirst 10 eligible entities:")
for eid, rel_count in eligible_entities[:10]:
    print(f"  Entity {eid}: {rel_count} unique relationships")
