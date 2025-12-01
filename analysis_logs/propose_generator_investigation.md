# Propose Generator Investigation - Why Only 5 Outputs for Entity Graph

**Date:** 2025-12-01
**Issue:** propose_generator only generates 5/5 outputs for entity graph

---

## Investigation Summary

The propose_generator produces exactly **5 outputs** because only **5 entities are eligible** after the relationship deduplication logic.

---

## Root Cause Analysis

### The Deduplication Logic

Located in `src/components/propose_generator.py:171-182`:

```python
already_have_chunks = set()
objective_relationships = []
for relationship_item in entity_relationship_graph['relationships']:
    chunk_id = relationship_item['id']  # <-- KEY: Deduplicating by chunk ID
    if chunk_id in already_have_chunks:
        continue
    already_have_chunks.add(chunk_id)
    objective_relationships.append(relationship_item)
```

**The code deduplicates relationships by their `id` field (chunk/document ID), NOT by relationship content!**

### The Filtering Condition

At `src/components/propose_generator.py:290-291`:

```python
if entity_relationship_prompt == "" or len(cur_objective_relationship_prompts) <= 1:
    continue  # Skip entities with ≤1 unique chunk relationships
```

---

## The Numbers

### Entity Distribution by Chunk Coverage

```
Distribution of entities by number of unique chunks their relationships span:
  1 chunk(s): 62 entities
  2 chunk(s): 5 entities

Entities with relationships in ≤1 chunk: 62
Entities with relationships in >1 chunks (ELIGIBLE): 5
```

### The 5 Eligible Entities

Only these 5 entities have relationships spanning 2 different chunks:
- Entity 15: relationships span 2 chunks
- Entity 18: relationships span 2 chunks
- Entity 24: relationships span 2 chunks
- Entity 27: relationships span 2 chunks
- Entity 48: relationships span 2 chunks

### Example: Entity 48

- **Total relationships in subgraph:** 7 relationships
- **Unique chunks those relationships come from:** 2 chunks (`doc1_question6` and `doc1_question8`)
- **After deduplication by chunk ID:** Only 2 relationships remain (one per chunk)
- **Result:** Passes the ≤1 filter and gets processed

```
All chunk IDs: ['doc1_question6', 'doc1_question6', 'doc1_question6',
                 'doc1_question6', 'doc1_question8', 'doc1_question8',
                 'doc1_question8']
Unique chunk IDs: ['doc1_question6', 'doc1_question8']
Unique relationships after dedup: 2
```

---

## Relationship Count vs Chunk Count

There's a critical difference between:

1. **Total relationship count** (how many edges an entity has in the graph)
2. **Unique chunk count** (how many different source documents those relationships span)

### Initial Analysis (Misleading)

```
Entities with >1 relationships: 39 entities (58.2%)
```

This counted total relationships, which is NOT what the code filters on.

### Actual Filter Criteria (Correct)

```
Entities with >1 unique chunks: 5 entities (7.5%)
```

This is what actually determines eligibility.

---

## Why This Design?

The deduplication by chunk ID appears **intentional** - it ensures generated questions:
- Combine information from **multiple different source chunks**
- Are genuinely **multi-hop questions** requiring knowledge synthesis
- Don't just reword information from a single source

This is a strict requirement for multi-hop question generation quality.

---

## Data Files Generated During Investigation

1. `analyze_entity_graph.py` - Analyzes relationship count per entity
2. `debug_propose_generator.py` - Simulates the propose_generator logic (incomplete due to dependencies)
3. `analyze_deduplication.py` - Examines how relationships are deduplicated
4. `count_eligible_entities.py` - Counts entities by unique chunk coverage

---

## Key Code Locations

- `src/components/propose_generator.py:171-182` - Relationship deduplication logic
- `src/components/propose_generator.py:290-291` - Entity filtering condition
- `src/components/entity_graph_constructor.py:200-262` - Subgraph generation
- `src/prompts/propose_generator_entity_graph.txt` - Prompt template requiring multi-relationship questions

---

## Configuration Files Checked

- `multi_hop.env` - Missing `PROPOSE_GENERATOR_MAX_GEN_TIMES` (defaults to 300)
- `single_hop.env` - Has `PROPOSE_GENERATOR_MAX_GEN_TIMES=100`

---

## Conclusion

**The system is working as designed.**

Your dataset has only 5 entities with relationships spanning multiple chunks, therefore only 5 outputs are generated. This is not a bug but a reflection of:

1. How your data is structured (most entities only appear in single chunks)
2. The strict multi-hop requirement (relationships must span multiple chunks)

To get more outputs, you would need to either:
- Restructure your data so entities appear across multiple chunks
- Modify the deduplication logic to allow relationships from the same chunk
- Adjust the filtering criteria at line 290-291

However, any of these changes would affect the quality and nature of the multi-hop questions being generated.
