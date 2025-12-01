#!/usr/bin/env python3
"""
Count how many entities actually have >1 unique chunk IDs in their relationships
"""
import json

# Load the input data
with open('src/outputs/entity_eliminator.json', 'r') as f:
    data = json.load(f)

# Build a mapping of entity_id -> set of chunk IDs
entity_to_chunks = {}

for item in data:
    chunk_id = item['id']

    for rel in item.get('relationship', []):
        source_id = rel['source_entity_id']
        target_id = rel['target_entity_id']

        if source_id not in entity_to_chunks:
            entity_to_chunks[source_id] = set()
        if target_id not in entity_to_chunks:
            entity_to_chunks[target_id] = set()

        entity_to_chunks[source_id].add(chunk_id)
        entity_to_chunks[target_id].add(chunk_id)

# Count entities by unique chunk count
chunk_count_distribution = {}
for entity_id, chunk_set in entity_to_chunks.items():
    count = len(chunk_set)
    if count not in chunk_count_distribution:
        chunk_count_distribution[count] = 0
    chunk_count_distribution[count] += 1

print("Distribution of entities by number of unique chunks their relationships span:")
for chunk_count in sorted(chunk_count_distribution.keys()):
    print(f"  {chunk_count} chunk(s): {chunk_count_distribution[chunk_count]} entities")

entities_with_1_chunk = chunk_count_distribution.get(1, 0)
entities_with_2plus_chunks = sum(count for chunks, count in chunk_count_distribution.items() if chunks > 1)

print(f"\nEntities with relationships in ≤1 chunk: {entities_with_1_chunk}")
print(f"Entities with relationships in >1 chunks (ELIGIBLE): {entities_with_2plus_chunks}")

# Show which specific entities are eligible
eligible_entities = [(eid, len(chunks)) for eid, chunks in entity_to_chunks.items() if len(chunks) > 1]
eligible_entities.sort(key=lambda x: x[1], reverse=True)

print(f"\nEligible entity IDs (sorted by chunk count):")
for eid, chunk_count in eligible_entities[:20]:
    print(f"  Entity {eid}: relationships span {chunk_count} chunks")

print(f"\nTotal eligible entities: {len(eligible_entities)}")
