"""
Session segmentation: converts a raw stream of user interaction events into
discrete sessions using a time-gap threshold, then into per-session
item-transition graphs for SR-GNN. See Milestone 2 report, Section 4.
"""

import pandas as pd
from app.config import SESSION_GAP_MINUTES


def segment_into_sessions(interactions: pd.DataFrame, gap_minutes: int = SESSION_GAP_MINUTES) -> pd.DataFrame:
    """
    interactions: DataFrame with columns [user_id, item_id, timestamp], sorted
    by user_id then timestamp.

    Adds a `session_id` column: a new session starts whenever the gap since
    the previous interaction for that user exceeds `gap_minutes`.
    """
    df = interactions.sort_values(["user_id", "timestamp"]).copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df["gap"] = df.groupby("user_id")["timestamp"].diff()
    df["new_session"] = (df["gap"].isna()) | (df["gap"] > pd.Timedelta(minutes=gap_minutes))
    df["session_id"] = df.groupby("user_id")["new_session"].cumsum()
    df["session_id"] = df["user_id"].astype(str) + "_" + df["session_id"].astype(str)

    return df.drop(columns=["gap", "new_session"])


def build_session_graph(session_items: list[str]) -> dict:
    """
    Converts one session's ordered item sequence into a directed transition
    graph, as SR-GNN expects. Handles repeated clicks within a session by
    accumulating edge weights rather than creating duplicate edges.

    Returns {"nodes": [...unique items in order of first appearance...],
             "edges": {(from_idx, to_idx): weight}}
    """
    nodes = []
    node_index = {}
    for item in session_items:
        if item not in node_index:
            node_index[item] = len(nodes)
            nodes.append(item)

    edges: dict[tuple[int, int], int] = {}
    for a, b in zip(session_items, session_items[1:]):
        edge = (node_index[a], node_index[b])
        edges[edge] = edges.get(edge, 0) + 1

    return {"nodes": nodes, "edges": edges}
