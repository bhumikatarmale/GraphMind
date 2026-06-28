import os
import json
import networkx as nx
from pyvis.network import Network
from src import config
from src import llm

class GraphStore:
    def __init__(self):
        self.graph_path = os.path.join(config.GRAPH_DIR, "knowledge_graph.json")
        self.graph = nx.DiGraph()
        self.load()

    def load(self):
        """Loads the graph from a JSON file if it exists."""
        if os.path.exists(self.graph_path):
            try:
                with open(self.graph_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                self.graph = nx.DiGraph()
                for node in data.get("nodes", []):
                    self.graph.add_node(node["id"], **node.get("data", {}))
                for edge in data.get("edges", []):
                    self.graph.add_edge(edge["source"], edge["target"], **edge.get("data", {}))
                print(f"Loaded graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges.")
            except Exception as e:
                print(f"Error loading graph, initializing empty graph: {e}")
                self.graph = nx.DiGraph()
        else:
            self.graph = nx.DiGraph()

    def save(self):
        """Saves the graph to a JSON file."""
        data = {
            "nodes": [{"id": node, "data": self.graph.nodes[node]} for node in self.graph.nodes],
            "edges": [{"source": u, "target": v, "data": self.graph.edges[u, v]} for u, v in self.graph.edges]
        }
        try:
            with open(self.graph_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving graph: {e}")

    def add_relations_from_chunk(self, chunk: dict):
        """
        Uses LLM to extract entities and relations from a text chunk,
        and adds them to the NetworkX graph.
        """
        text = chunk["text"]
        source = chunk["source"]
        page = chunk["page"]
        
        # Prepare extraction prompt
        prompt = config.ENTITY_EXTRACTION_PROMPT.format(text=text)
        
        try:
            triples = llm.generate_json(prompt)
            if not isinstance(triples, list):
                # If LLM didn't return a list, skip
                return
                
            for triple in triples:
                if not isinstance(triple, dict):
                    continue
                subj = triple.get("subject")
                rel = triple.get("relation")
                obj = triple.get("object")
                
                if not subj or not rel or not obj:
                    continue
                
                # Normalize names: strip, title case to merge synonyms
                subj = str(subj).strip().title()
                obj = str(obj).strip().title()
                rel = str(rel).strip().lower()
                
                # Add nodes if they don't exist
                if not self.graph.has_node(subj):
                    self.graph.add_node(subj, type="Entity", degree=0)
                if not self.graph.has_node(obj):
                    self.graph.add_node(obj, type="Entity", degree=0)
                
                # Add or update edge
                if self.graph.has_edge(subj, obj):
                    edge_data = self.graph.edges[subj, obj]
                    # Update sources list and count
                    if source not in edge_data.get("sources", []):
                        edge_data["sources"].append(source)
                    edge_data["count"] = edge_data.get("count", 1) + 1
                    # Append page if not already listed
                    page_str = f"{source}:p{page}"
                    if page_str not in edge_data.get("pages", []):
                        edge_data["pages"].append(page_str)
                else:
                    self.graph.add_edge(
                        subj, 
                        obj, 
                        relation=rel, 
                        sources=[source], 
                        pages=[f"{source}:p{page}"],
                        count=1
                    )
            
            # Update degree attribute for visualization sizing
            for node in self.graph.nodes:
                self.graph.nodes[node]["degree"] = self.graph.degree(node)
                
        except Exception as e:
            print(f"Error extracting relations from chunk: {e}")

    def traverse_subgraph(self, seed_entities: list[str], max_depth: int = 1) -> list[dict]:
        """
        Traverses the graph starting from seed entities up to max_depth,
        returning a list of relations (edges) found.
        """
        visited_nodes = set()
        retrieved_relations = []
        
        # Normalize seed entities
        normalized_seeds = [seed.strip().title() for seed in seed_entities]
        
        # Queue format: (node, depth)
        queue = [(node, 0) for node in normalized_seeds if self.graph.has_node(node)]
        
        for node, _ in queue:
            visited_nodes.add(node)
            
        # Perform BFS up to max_depth
        current_queue = list(queue)
        next_queue = []
        
        for depth in range(max_depth):
            for node, d in current_queue:
                # Find all neighbors (incoming and outgoing)
                # Outgoing edges
                for neighbor in self.graph.successors(node):
                    edge_data = self.graph.edges[node, neighbor]
                    rel_dict = {
                        "subject": node,
                        "relation": edge_data.get("relation", "connected_to"),
                        "object": neighbor,
                        "sources": edge_data.get("sources", []),
                        "pages": edge_data.get("pages", []),
                        "count": edge_data.get("count", 1)
                    }
                    if rel_dict not in retrieved_relations:
                        retrieved_relations.append(rel_dict)
                    if neighbor not in visited_nodes:
                        visited_nodes.add(neighbor)
                        next_queue.append((neighbor, d + 1))
                # Incoming edges
                for predecessor in self.graph.predecessors(node):
                    edge_data = self.graph.edges[predecessor, node]
                    rel_dict = {
                        "subject": predecessor,
                        "relation": edge_data.get("relation", "connected_to"),
                        "object": node,
                        "sources": edge_data.get("sources", []),
                        "pages": edge_data.get("pages", []),
                        "count": edge_data.get("count", 1)
                    }
                    if rel_dict not in retrieved_relations:
                        retrieved_relations.append(rel_dict)
                    if predecessor not in visited_nodes:
                        visited_nodes.add(predecessor)
                        next_queue.append((predecessor, d + 1))
            current_queue = next_queue
            next_queue = []
            
        return retrieved_relations

    def find_seeds_in_query(self, query: str) -> list[str]:
        """Matches nodes in the graph against words/phrases in the user's query."""
        seeds = []
        query_lower = query.lower()
        
        # Simple string-matching: see if any node name is contained in the query
        for node in self.graph.nodes:
            if node.lower() in query_lower:
                seeds.append(node)
        return seeds

    def generate_visualization_html(self, output_filename: str = "graph.html") -> str:
        """
        Generates an interactive Pyvis HTML visualization of the knowledge graph.
        Returns the absolute path to the HTML file.
        """
        net = Network(
            height="500px", 
            width="100%", 
            bgcolor="#1e1e1e",  # Dark mode background
            font_color="#ffffff",
            directed=True,
            notebook=False
        )
        
        # Apply physics options for premium layout feel
        net.set_options("""
        var options = {
          "physics": {
            "forceAtlas2Based": {
              "gravitationalConstant": -60,
              "centralGravity": 0.015,
              "springLength": 120,
              "springConstant": 0.08
            },
            "maxVelocity": 50,
            "solver": "forceAtlas2Based",
            "timestep": 0.35,
            "stabilization": { "iterations": 150 }
          },
          "edges": {
            "smooth": {
              "type": "continuous",
              "forceDirection": "none"
            }
          }
        }
        """)

        # Add nodes with deg-based sizing and custom colors
        for node in self.graph.nodes:
            deg = self.graph.nodes[node].get("degree", 1)
            size = 15 + min(deg * 3, 30)
            
            # Node color: Cyan for high connection, Slate Blue for low connection
            color = "#00adb5" if deg > 2 else "#393e46"
            
            net.add_node(
                node, 
                label=node, 
                title=f"Entity: {node}\nDegree: {deg}", 
                size=size,
                color=color
            )
            
        # Add edges
        for u, v in self.graph.edges:
            edge_data = self.graph.edges[u, v]
            relation = edge_data.get("relation", "")
            sources = ", ".join(edge_data.get("sources", []))
            
            net.add_edge(
                u, 
                v, 
                label=relation, 
                title=f"Relation: {relation}\nSources: {sources}",
                color="#00adb5",
                width=1.5
            )
            
        # Save HTML file
        output_path = os.path.join(config.GRAPH_DIR, output_filename)
        net.save_graph(output_path)
        return output_path

    def reset(self):
        """Clears the graph."""
        self.graph = nx.DiGraph()
        if os.path.exists(self.graph_path):
            try:
                os.remove(self.graph_path)
            except Exception:
                pass
        self.save()
