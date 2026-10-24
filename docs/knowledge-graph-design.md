# Knowledge graph status and next design boundary

## Implemented graph

The current graph is a provenance view built from indexed, privacy-versioned chunks. It contains document, physical PDF page and passage nodes with `CONTAINS` relationships. Each passage preserves its source document ID, filename, page, chunk ID and sanitized text. `GET /api/knowledge-graph` returns this bounded representation; old indexes are not exposed before explicit re-indexing.

## Not implemented

The graph does not extract regulations, obligations, entities, roles, deadlines, thresholds, scopes, exceptions, supersession or potential-conflict edges. It does not use Neo4j or perform graph traversal for question answering. The current graph must therefore be described as a source-provenance graph, not a regulatory knowledge graph.

## Safe extension requirements

Any regulatory edge must be derived from sanitized chunk text and retain an exact, validated source quotation plus document hash, page and chunk provenance. Invalid quotations must remain unverified and must not be returned as evidence. Traversal must be bounded, and graph failures must be reported as unavailable rather than as successful empty results. Extracted edges are source representations, not authoritative legal conclusions. Supersession or authority requires an explicitly verified hierarchy and effective-date policy.

No graph-accuracy or graph-cost benchmark has been run. See `docs/feature-evaluation-report.md`.
