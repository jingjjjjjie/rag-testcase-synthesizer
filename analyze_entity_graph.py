#!/usr/bin/env python3
"""
Analyze the entity graph to determine how many entities have >1 relationship
"""
import json
from collections import defaultdict

# Load the entity eliminator output
with open('src/outputs/entity_eliminator.json', 'r') as f:
    data = json.load(f)

# Count relationships per entity
entity_relationship_count = defaultdict(int)

for item in data:
    if 'relationship' in item:
        for rel in item['relationship']:
            source_id = rel['source_entity_id']
            target_id = rel['target_entity_id']
            entity_relationship_count[source_id] += 1
            entity_relationship_count[target_id] += 1

# Analyze the distribution
total_entities = len(entity_relationship_count)
entities_with_0_rel = sum(1 for count in entity_relationship_count.values() if count == 0)
entities_with_1_rel = sum(1 for count in entity_relationship_count.values() if count == 1)
entities_with_2_plus_rel = sum(1 for count in entity_relationship_count.values() if count > 1)

print(f"Total entities: {total_entities}")
print(f"Entities with 0 relationships: {entities_with_0_rel}")
print(f"Entities with 1 relationship: {entities_with_1_rel}")
print(f"Entities with >1 relationships: {entities_with_2_plus_rel}")
print(f"\nPercentage with >1 relationships: {entities_with_2_plus_rel/total_entities*100:.1f}%")

# Show the distribution
relationship_distribution = defaultdict(int)
for count in entity_relationship_count.values():
    relationship_distribution[count] += 1

print(f"\nRelationship distribution:")
for rel_count in sorted(relationship_distribution.keys()):
    print(f"  {rel_count} relationship(s): {relationship_distribution[rel_count]} entities")

# Show which entities have already been processed
with open('src/outputs/propose_generator_entity_graph.json', 'r') as f:
    processed = json.load(f)

processed_ids = set(int(k) for k in processed.keys())
print(f"\nAlready processed entities: {len(processed_ids)}")
print(f"Processed entity IDs: {sorted(processed_ids)}")

# Find entities eligible but not yet processed
eligible_entities = [eid for eid, count in entity_relationship_count.items() if count > 1]
not_yet_processed = [eid for eid in eligible_entities if eid not in processed_ids]

print(f"\nEligible entities (>1 relationships): {len(eligible_entities)}")
print(f"Not yet processed: {len(not_yet_processed)}")
if len(not_yet_processed) <= 20:
    print(f"Not yet processed entity IDs: {sorted(not_yet_processed)}")
