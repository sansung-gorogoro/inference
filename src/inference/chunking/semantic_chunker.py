"""Semantic chunk assembly pass (opt-in, between Pass 1 and Pass 2).

This module re-groups sentences within each silence-bounded region
(IntermediateChunk from Pass 1) by semantic coherence, using BGE-M3
sentence embeddings and centroid-vs-next cosine similarity.

Algorithm (centroid_v1):
    1. Flatten all IntermediateChunks into sentences (preserving region provenance).
    2. Embed ALL sentences in one batched BGE-M3 call.
    3. Compute adjacent-sentence similarity series across the full lecture.
    4. Derive adaptive threshold T_adj from the quantile of that distribution.
    5. Within each silence-bounded region, assemble new IntermediateChunks:
       - Start with a warmup of min_sentences.
       - After warmup, split when cosine(centroid, next_sentence) < T_adj.
    6. Output new IntermediateChunks (fed into existing Pass 2).

Silence boundaries are hard: semantic assembly never merges across regions.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

from inference.chunking.silence_chunker import IntermediateChunk

logger = logging.getLogger(__name__)

# Fallback threshold when too few adjacent pairs for quantile estimation
_T_ADJ_FALLBACK = 0.75
_MIN_ADJ_PAIRS_FOR_QUANTILE = 10


# ---------------------------------------------------------------------------
# Vector utilities (cosine similarity, centroid, normalization)
# ---------------------------------------------------------------------------


def _l2_norm(vec: list[float]) -> float:
    """Compute L2 norm of a vector.

    Args:
        vec: Input vector.

    Returns:
        L2 norm (>= 0.0). Returns 0.0 for zero vector.
    """
    s = 0.0
    for v in vec:
        s += v * v
    return math.sqrt(s)


def _normalize(vec: list[float]) -> list[float]:
    """L2-normalize a vector in-place-safe (returns new list).

    Args:
        vec: Input vector.

    Returns:
        Unit vector. Returns zero vector if input norm is ~0.
    """
    norm = _l2_norm(vec)
    if norm < 1e-12:
        return [0.0] * len(vec)
    inv = 1.0 / norm
    return [v * inv for v in vec]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors.

    Both vectors are L2-normalized before dot product.

    Args:
        a: First vector.
        b: Second vector (must be same dimensionality as *a*).

    Returns:
        Cosine similarity in [-1.0, 1.0].
    """
    na = _normalize(a)
    nb = _normalize(b)
    dot = 0.0
    for x, y in zip(na, nb):
        dot += x * y
    return dot


def _centroid(embeddings: list[list[float]]) -> list[float]:
    """Compute mean (centroid) of a list of embedding vectors.

    Args:
        embeddings: Non-empty list of equal-length vectors.

    Returns:
        Element-wise mean vector (NOT L2-normalized).
    """
    dim = len(embeddings[0])
    acc = [0.0] * dim
    for emb in embeddings:
        for i, v in enumerate(emb):
            acc[i] += v
    n = len(embeddings)
    return [a / n for a in acc]


# ---------------------------------------------------------------------------
# Quantile helper (pure-python, no numpy)
# ---------------------------------------------------------------------------


def _nearest_rank_quantile(values: list[float], q: float) -> float:
    """Compute nearest-rank quantile (deterministic, no interpolation).

    Args:
        values: Non-empty list of floats.
        q: Quantile in [0.0, 1.0].

    Returns:
        The q-th quantile value.
    """
    sorted_vals = sorted(values)
    k = math.ceil(q * len(sorted_vals)) - 1
    k = max(0, min(k, len(sorted_vals) - 1))
    return sorted_vals[k]


# ---------------------------------------------------------------------------
# Sentence splitting helper (reuses wtpsplit from sentence_chunker)
# ---------------------------------------------------------------------------


@dataclass
class _SentenceInfo:
    """Internal sentence metadata for semantic assembly."""

    text: str
    region_idx: int  # index of the parent IntermediateChunk
    sentence_idx: int  # index within the region
    # Proportional timestamp anchoring within the parent region
    start_time: float
    end_time: float


def _split_into_sentences(
    chunks: list[IntermediateChunk],
) -> list[_SentenceInfo]:
    """Split all IntermediateChunks into sentences with provenance.

    Uses the same SentenceSplitter from sentence_chunker to ensure
    consistent sentence boundary detection.

    Args:
        chunks: List of silence-bounded IntermediateChunks.

    Returns:
        Flat list of _SentenceInfo with region provenance and timestamps.
    """
    from inference.chunking.sentence_chunker import SentenceSplitter

    splitter = SentenceSplitter()
    sentences: list[_SentenceInfo] = []

    for region_idx, chunk in enumerate(chunks):
        raw_sentences = splitter.split(chunk.text)
        if not raw_sentences:
            # Edge case: splitter returned nothing — treat whole text as one sentence
            raw_sentences = [chunk.text]

        # Filter empty strings
        raw_sentences = [s.strip() for s in raw_sentences if s.strip()]
        if not raw_sentences:
            continue

        # Proportional timestamp allocation within the region
        total_chars = sum(len(s) for s in raw_sentences)
        duration = chunk.end_time - chunk.start_time
        if total_chars == 0:
            # Degenerate: all empty — skip
            continue

        char_offset = 0
        for sent_idx, sent_text in enumerate(raw_sentences):
            frac_start = char_offset / total_chars
            frac_end = (char_offset + len(sent_text)) / total_chars
            s_start = chunk.start_time + frac_start * duration
            s_end = chunk.start_time + frac_end * duration

            # Clamp to parent region bounds
            s_start = max(chunk.start_time, min(s_start, chunk.end_time))
            s_end = max(s_start + 0.001, min(s_end, chunk.end_time))

            sentences.append(
                _SentenceInfo(
                    text=sent_text,
                    region_idx=region_idx,
                    sentence_idx=sent_idx,
                    start_time=round(s_start, 3),
                    end_time=round(s_end, 3),
                )
            )
            char_offset += len(sent_text)

    return sentences


# ---------------------------------------------------------------------------
# Main semantic assembly function
# ---------------------------------------------------------------------------


def semantic_assembly(
    chunks: list[IntermediateChunk],
    adj_quantile: float = 0.2,
    sim_clamp_min: float = 0.35,
    sim_clamp_max: float = 0.90,
    min_sentences: int = 3,
) -> list[IntermediateChunk]:
    """Re-group sentences within silence-bounded regions by semantic coherence.

    This is the "Pass 1.5" step inserted between silence-gap chunking (Pass 1)
    and sentence-boundary/token-limit chunking (Pass 2).

    Args:
        chunks: IntermediateChunks from Pass 1 (silence-gap grouping).
        adj_quantile: Quantile of adjacent similarity distribution for threshold.
        sim_clamp_min: Minimum allowed adaptive threshold.
        sim_clamp_max: Maximum allowed adaptive threshold.
        min_sentences: Warmup — first N sentences always stay in initial chunk.

    Returns:
        New list of IntermediateChunks, semantically re-grouped within
        each original silence-bounded region. Silence boundaries are preserved.
    """
    if not chunks:
        return []

    logger.info(
        "Starting semantic assembly",
        extra={
            "input_regions": len(chunks),
            "adj_quantile": adj_quantile,
            "sim_clamp_min": sim_clamp_min,
            "sim_clamp_max": sim_clamp_max,
            "min_sentences": min_sentences,
        },
    )

    # Step 1: Split all regions into sentences
    all_sentences = _split_into_sentences(chunks)
    if not all_sentences:
        logger.warning("No sentences extracted, returning original chunks")
        return chunks

    logger.info(f"Extracted {len(all_sentences)} sentences from {len(chunks)} regions")

    # Step 2: Embed all sentences in one batch
    from inference.embedding.bge_embedder import BGEEmbedder

    embedder = BGEEmbedder()
    texts = [s.text for s in all_sentences]
    embeddings = embedder.embed_texts(texts)

    # Step 3: Compute adjacent similarities across the full lecture
    adj_sims: list[float] = []
    for i in range(len(embeddings) - 1):
        sim = cosine_similarity(embeddings[i], embeddings[i + 1])
        adj_sims.append(sim)

    # Step 4: Derive adaptive threshold
    if len(adj_sims) >= _MIN_ADJ_PAIRS_FOR_QUANTILE:
        raw_threshold = _nearest_rank_quantile(adj_sims, adj_quantile)
        t_adj = max(sim_clamp_min, min(raw_threshold, sim_clamp_max))
        threshold_mode = "adaptive"
    else:
        t_adj = _T_ADJ_FALLBACK
        threshold_mode = "fallback"

    logger.info(
        f"Semantic threshold resolved: T_adj={t_adj:.4f} (mode={threshold_mode})",
        extra={
            "threshold": t_adj,
            "threshold_mode": threshold_mode,
            "adj_pairs": len(adj_sims),
            "adj_quantile": adj_quantile,
        },
    )

    # Step 5: Assemble new IntermediateChunks within each region
    # Group sentences by region
    region_sentences: dict[int, list[int]] = {}  # region_idx -> [global_sentence_idx]
    for global_idx, sent in enumerate(all_sentences):
        region_sentences.setdefault(sent.region_idx, []).append(global_idx)

    output_chunks: list[IntermediateChunk] = []

    for region_idx in sorted(region_sentences.keys()):
        sent_indices = region_sentences[region_idx]

        if len(sent_indices) <= 1:
            # Single sentence region — emit as-is
            s = all_sentences[sent_indices[0]]
            output_chunks.append(
                IntermediateChunk(
                    text=s.text,
                    start_time=s.start_time,
                    end_time=s.end_time,
                )
            )
            continue

        # Assembly within this region
        current_sent_indices: list[int] = [sent_indices[0]]
        current_embeddings: list[list[float]] = [embeddings[sent_indices[0]]]

        for pos in range(1, len(sent_indices)):
            g_idx = sent_indices[pos]
            next_emb = embeddings[g_idx]

            # Warmup: always append first min_sentences
            if len(current_sent_indices) < min_sentences:
                current_sent_indices.append(g_idx)
                current_embeddings.append(next_emb)
                continue

            # Compute centroid of current chunk and similarity to next sentence
            cent = _centroid(current_embeddings)
            sim = cosine_similarity(cent, next_emb)

            if sim < t_adj:
                # Split: finalize current chunk
                _emit_chunk(all_sentences, current_sent_indices, output_chunks)

                # Start new chunk with the next sentence
                current_sent_indices = [g_idx]
                current_embeddings = [next_emb]
            else:
                # Continue current chunk
                current_sent_indices.append(g_idx)
                current_embeddings.append(next_emb)

        # Finalize last chunk in region
        if current_sent_indices:
            _emit_chunk(all_sentences, current_sent_indices, output_chunks)

    logger.info(
        "Semantic assembly completed",
        extra={
            "input_regions": len(chunks),
            "output_chunks": len(output_chunks),
            "threshold": t_adj,
            "threshold_mode": threshold_mode,
        },
    )

    return output_chunks


def _emit_chunk(
    all_sentences: list[_SentenceInfo],
    sent_indices: list[int],
    output: list[IntermediateChunk],
) -> None:
    """Create an IntermediateChunk from a group of sentence indices and append to output.

    Args:
        all_sentences: Full sentence info list.
        sent_indices: Indices into all_sentences for this chunk.
        output: Output list to append the new chunk to.
    """
    texts = [all_sentences[i].text for i in sent_indices]
    combined_text = " ".join(texts).strip()
    if not combined_text:
        return

    start_time = all_sentences[sent_indices[0]].start_time
    end_time = all_sentences[sent_indices[-1]].end_time

    # Ensure valid timestamps
    if end_time <= start_time:
        end_time = start_time + 0.001

    output.append(
        IntermediateChunk(
            text=combined_text,
            start_time=start_time,
            end_time=end_time,
        )
    )
