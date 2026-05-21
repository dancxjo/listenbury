# Riper evaluation: Link Grammar-style sentence analysis

Follow-up mapping table and implementation priorities:
[`riper-link-grammar-mapping.md`](./riper-link-grammar-mapping.md).

## Why evaluate a link model

Riper prosody decisions often depend on local typed relationships (`to` + infinitive, vocative boundaries, contrast pairs, comma-island behavior) rather than a single global constituency tree.  
A Link Grammar-inspired representation is a good fit for that requirement: it models a sentence as compatible labeled links between word pairs, and allows multiple analyses to coexist.

## Model inventory (adapted architecture, not code reuse)

- Words expose connector expectations (e.g., infinitival marker, determiner-noun, auxiliary-verb).
- Compatible connectors form labeled word-pair links.
- A parse is a graph of links, not only a tree.
- Grammaticality is approximated by satisfying enough compatible constraints.
- Multiple parses can be retained for ambiguous phrases.
- Parse ranking/scoring can be attached to each candidate link graph.

Riper implementation note: this repository uses a small internal `SyntacticLink` graph and emits symbolic `AnalysisClaim`s. It does **not** depend on AbiWord internals or Link Grammar dictionary formats.

## Citations for reimplementation study

1. Daniel D. Sleator and Davy Temperley, **Parsing English with a Link Grammar** (Third International Workshop on Parsing Technologies, 1993).  
   ACL Anthology: https://aclanthology.org/1993.iwpt-1.28/
2. Daniel D. Sleator and Davy Temperley, **Parsing English with a Link Grammar** (Carnegie Mellon University Technical Report, 1991).  
   CMU technical report record: https://www.cs.cmu.edu/~sleator/papers/link-grammar.pdf
3. Davy Temperley, John Lafferty, and Daniel Sleator, **Link Grammar: A Parsimony-Based Grammar Formalism for Natural Languages** (1999).  
   Project bibliography entry: https://www.link.cs.cmu.edu/link/

These references are sufficient to reimplement compatible *ideas* (connector compatibility, link labeling, ambiguity retention, and ranking) without copying external source code.
