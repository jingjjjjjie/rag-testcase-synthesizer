#!/usr/bin/env python3
"""
Analyze the deduplication logic that might be filtering out relationships
"""
import json

# Load one of the processed outputs to see what it looks like
with open('src/outputs/propose_generator_entity_graph.json', 'r') as f:
    outputs = json.load(f)

# Let's examine the relationship deduplication
for entity_id, entity_data in outputs.items():
    relationships = entity_data['relationships']

    print(f"\nEntity {entity_id}:")
    print(f"  Total relationships in subgraph: {len(relationships)}")

    # Simulate the deduplication logic from propose_generator.py line 171-182
    already_have_chunks = set()
    unique_relationships = []
    duplicates = []

    for relationship_item in relationships:
        chunk_id = relationship_item['id']
        if chunk_id in already_have_chunks:
            duplicates.append(chunk_id)
            continue
        already_have_chunks.add(chunk_id)
        unique_relationships.append(relationship_item)

    print(f"  Unique relationships (after dedup by 'id'): {len(unique_relationships)}")
    print(f"  Duplicates removed: {len(duplicates)}")

    if len(duplicates) > 0:
        print(f"  Duplicate chunk IDs: {duplicates}")

    # Show all chunk IDs
    all_chunk_ids = [r['id'] for r in relationships]
    print(f"  All chunk IDs: {all_chunk_ids}")
    print(f"  Unique chunk IDs: {list(already_have_chunks)}")

    # This is the condition that determines if entity is processed
    would_be_skipped = len(unique_relationships) <= 1
    print(f"  Would be skipped (≤1 unique relationships): {would_be_skipped}")
