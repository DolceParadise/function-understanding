# Why `jinaai/jina-embeddings-v2-base-code`

This project turns C function datapoints into vectors for semantic retrieval and analysis, so the embedding model needs to be code-aware, compact, and stable for downstream cosine-similarity workflows.

`jinaai/jina-embeddings-v2-base-code` fits that use case well because it is trained specifically for code and technical text, not just general natural language. It supports long inputs, which matters when a function body is large, and it is designed for tasks like code search, similarity matching, clustering, and classification. That makes it a good default for the later workflows:

- function similarity search
- retrieval of similar functions for a query function
- clustering by semantic purpose
- lightweight classification tasks such as side effect prediction

It is also a practical size for offline indexing. The model is a 161M-parameter code embedding model, which is small enough to run locally without needing a large embedding service, while still being specialized enough to outperform generic text embeddings on code-centric tasks.

For this repository, the important implementation detail is that the embedding input should be the function-centric text itself, not the labels. That keeps the vectors useful for future retrieval and classification experiments instead of leaking supervision into the representation.
