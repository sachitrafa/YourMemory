"""
Graph visualization endpoint for YourMemory demo.

Usage:
    http://localhost:3033/graph?memoryId=364&userId=sachit
"""

import os
import pickle
import json
from fastapi import APIRouter, Query
from fastapi.responses import HTMLResponse, JSONResponse

router = APIRouter()


def get_graph_data(memory_id: int, user_id: str, depth: int = 2):
    """
    Extract subgraph around a memory node for visualization.
    Returns nodes and edges in Cytoscape.js format.
    """
    # Use graph path matching current DB
    db_path = os.getenv("YOURMEMORY_DB", os.path.expanduser("~/.yourmemory/memories.duckdb"))
    if "demo.duckdb" in db_path:
        graph_path = os.path.expanduser("~/.yourmemory/demo_graph.pkl")
    else:
        graph_path = os.path.expanduser("~/.yourmemory/graph.pkl")

    if not os.path.exists(graph_path):
        return {"nodes": [], "edges": []}

    import networkx as nx
    G = pickle.load(open(graph_path, "rb"))

    if memory_id not in G:
        return {"nodes": [], "edges": []}

    # BFS to depth N from seed
    visited = {memory_id}
    frontier = [(memory_id, 0)]
    nodes_to_include = {memory_id}

    while frontier:
        node, dist = frontier.pop(0)
        if dist >= depth:
            continue
        for nbr in G.neighbors(node):
            if nbr not in visited:
                visited.add(nbr)
                nodes_to_include.add(nbr)
                frontier.append((nbr, dist + 1))

    # Fetch memory content from DB
    from src.db.connection import get_conn
    conn = get_conn()
    rows = conn.execute(
        f"SELECT id, content, category FROM memories WHERE id IN ({','.join(map(str, nodes_to_include))})"
    ).fetchall()
    conn.close()

    content_map = {r[0]: {"content": r[1], "category": r[2]} for r in rows}

    # Build Cytoscape.js elements
    nodes = []
    for nid in nodes_to_include:
        info = content_map.get(nid, {"content": f"Memory {nid}", "category": "unknown"})
        nodes.append({
            "data": {
                "id": str(nid),
                "label": info["content"][:40] + ("..." if len(info["content"]) > 40 else ""),
                "category": info["category"],
                "isRoot": (nid == memory_id),
            }
        })

    edges = []
    for src in nodes_to_include:
        for tgt in G.neighbors(src):
            if tgt in nodes_to_include:
                edge_data = G[src][tgt]
                weight = edge_data.get("weight", 0.5)
                edges.append({
                    "data": {
                        "source": str(src),
                        "target": str(tgt),
                        "weight": round(weight, 3),
                    }
                })

    return {"nodes": nodes, "edges": edges}


@router.get("/graph/data")
def graph_data(
    memoryId: int = Query(..., description="Center node memory ID"),
    userId: str = Query("sachit", description="User ID"),
    depth: int = Query(2, ge=1, le=3, description="BFS depth"),
):
    """JSON API: returns graph data for a memory."""
    return JSONResponse(get_graph_data(memoryId, userId, depth))


@router.get("/graph")
def graph_viz(
    memoryId: int = Query(..., description="Center node memory ID"),
    userId: str = Query("sachit", description="User ID"),
    depth: int = Query(2, ge=1, le=3, description="BFS depth"),
):
    """Interactive graph visualization UI."""
    return HTMLResponse(content=_GRAPH_HTML.replace("MEMORY_ID_PLACEHOLDER", str(memoryId)))


_GRAPH_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>YourMemory Graph Visualization</title>
  <script src="https://unpkg.com/cytoscape@3.28.1/dist/cytoscape.min.js"></script>
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body { font-family: system-ui, -apple-system, sans-serif; background: #0d1117; color: #e6edf3; }
    #cy { width: 100vw; height: 100vh; }
    #info {
      position: fixed; top: 20px; left: 20px; background: #161b22; border: 1px solid #30363d;
      border-radius: 8px; padding: 16px; max-width: 320px; font-size: 14px; line-height: 1.6;
      box-shadow: 0 8px 24px rgba(0,0,0,0.5);
    }
    #info h3 { font-size: 16px; margin-bottom: 8px; color: #00D4FF; }
    #info .label { font-weight: 600; color: #8b949e; margin-top: 8px; }
    .badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; text-transform: uppercase; }
    .badge.fact { background: #1f6feb22; color: #58a6ff; }
    .badge.strategy { background: #3fb95022; color: #3fb950; }
    .badge.failure { background: #f8514922; color: #f85149; }
    .badge.assumption { background: #d2a8ff22; color: #d2a8ff; }
  </style>
</head>
<body>
  <div id="cy"></div>
  <div id="info">
    <h3>Graph Visualization</h3>
    <div class="label">Root Memory</div>
    <div id="rootLabel">Loading...</div>
    <div class="label" style="margin-top: 12px;">Instructions</div>
    <div style="color: #8b949e; font-size: 12px;">
      • Click nodes to see content<br>
      • Drag to reposition<br>
      • Scroll to zoom<br>
      • Edges = semantic similarity
    </div>
  </div>

  <script>
    const urlParams = new URLSearchParams(window.location.search);
    const memoryId = urlParams.get('memoryId') || 'MEMORY_ID_PLACEHOLDER';
    const userId = urlParams.get('userId') || 'sachit';
    const depth = parseInt(urlParams.get('depth')) || 2;

    fetch(`/graph/data?memoryId=${memoryId}&userId=${userId}&depth=${depth}`)
      .then(r => r.json())
      .then(data => {
        const cy = cytoscape({
          container: document.getElementById('cy'),
          elements: [...data.nodes, ...data.edges],
          style: [
            {
              selector: 'node',
              style: {
                'background-color': function(ele) {
                  if (ele.data('isRoot')) return '#00D4FF';
                  const cat = ele.data('category');
                  if (cat === 'strategy') return '#3fb950';
                  if (cat === 'failure') return '#f85149';
                  if (cat === 'assumption') return '#d2a8ff';
                  return '#58a6ff';
                },
                'label': 'data(label)',
                'width': function(ele) { return ele.data('isRoot') ? 60 : 40; },
                'height': function(ele) { return ele.data('isRoot') ? 60 : 40; },
                'font-size': '11px',
                'text-valign': 'bottom',
                'text-margin-y': 8,
                'text-wrap': 'wrap',
                'text-max-width': '120px',
                'color': '#e6edf3',
                'border-width': function(ele) { return ele.data('isRoot') ? 3 : 1; },
                'border-color': function(ele) { return ele.data('isRoot') ? '#00D4FF' : '#30363d'; },
              }
            },
            {
              selector: 'edge',
              style: {
                'width': function(ele) { return Math.max(1, ele.data('weight') * 4); },
                'line-color': '#30363d',
                'target-arrow-color': '#30363d',
                'target-arrow-shape': 'triangle',
                'curve-style': 'bezier',
                'opacity': 0.6,
              }
            },
            {
              selector: 'node:selected',
              style: {
                'border-width': 3,
                'border-color': '#00D4FF',
              }
            }
          ],
          layout: {
            name: 'cose',
            animate: true,
            animationDuration: 500,
            nodeRepulsion: 8000,
            idealEdgeLength: 100,
          }
        });

        // Show root label
        const rootNode = data.nodes.find(n => n.data.isRoot);
        if (rootNode) {
          const cat = rootNode.data.category;
          document.getElementById('rootLabel').innerHTML =
            `<span class="badge ${cat}">${cat}</span><br><span style="color:#c9d1d9; font-size:13px; margin-top:4px; display:block;">${rootNode.data.label}</span>`;
        }

        // Click handler
        cy.on('tap', 'node', function(evt) {
          const node = evt.target;
          const info = document.getElementById('info');
          const cat = node.data('category');
          info.innerHTML = `
            <h3>Memory #${node.data('id')}</h3>
            <span class="badge ${cat}">${cat}</span>
            <div class="label">Content</div>
            <div style="color:#c9d1d9; font-size:12px; margin-top:4px;">${node.data('label')}</div>
            ${node.data('isRoot') ? '<div style="margin-top:8px; color:#00D4FF; font-size:11px;">⬤ ROOT NODE</div>' : ''}
          `;
        });
      })
      .catch(err => {
        document.getElementById('info').innerHTML =
          `<h3 style="color:#f85149;">Error</h3><div style="color:#8b949e; font-size:12px;">${err.message}</div>`;
      });
  </script>
</body>
</html>
"""
