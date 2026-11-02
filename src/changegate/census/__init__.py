"""The census: the gate run over public derivatives of one base, summarized as a Drift Report.

The pipeline has four steps, one module each:

1. ``manifest``: list a base's public derivatives on the Hugging Face Hub (safetensors only),
   with license, downloads, last-modified date, declared purpose and a keyword-based
   "declares safety removal" flag (``declared``). The flag is a label taken from what the
   repository says about itself; it is never passed to the gate.
2. ``sample``: draw a fixed-seed sample per base, stratified by that flag.
3. ``runner``: download each sampled derivative to scratch, run the gate, record the results
   and delete the weights.
4. ``report``: aggregate the records into the Drift Report tables (``stats`` holds the
   interval arithmetic).

``hub`` is the only module that talks to the Hub; everything else takes a ``HubClient`` so
tests run on recorded responses. No language model is called anywhere in this package: the
flag is a regular expression, the purpose is the card's first paragraph, and every number
comes from the gate's own results.
"""
