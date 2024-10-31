import praw
import re
import networkx as nx
import csv
from collections import defaultdict

# Initialize PRAW with your credentials
reddit = praw.Reddit(
    client_id="XXX",
    client_secret="XXXX",
    user_agent="XXXX",
    username="XXX",
    password="XXXX"
)

# Function to collect posts and comments
def collect_data(subreddit_name, query, limit=100):
    subreddit = reddit.subreddit(subreddit_name)
    data = []
    for submission in subreddit.search(query, limit=limit):
        submission.comments.replace_more(limit=0)
        for comment in submission.comments.list():
            data.append({
                'author': comment.author.name if comment.author else 'deleted',
                'text': comment.body,
                'parent_id': comment.parent_id,
                'submission_id': submission.id,
                'subreddit': subreddit_name  # Add the subreddit field
            })
    return data

# Collect data from relevant subreddits
data_kl = collect_data('kendricklamar', 'Kendrick', limit=200)
data_drake = collect_data('Drizzy', 'Drake', limit=200)

# Combine the data from both subreddits
data = data_kl + data_drake

# Clean the text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

for entry in data:
    entry['text'] = clean_text(entry['text'])

# Construct the network
G = nx.Graph()

# Add nodes
for entry in data:
    author = entry['author']
    if author != 'deleted':
        G.add_node(author)

# Create edges based on thread participation and parent-child relationships
submission_dict = {}

# Group comments by submission
for entry in data:
    submission_id = entry['submission_id']
    if submission_id not in submission_dict:
        submission_dict[submission_id] = []
    submission_dict[submission_id].append(entry)

# Create edges for users in the same submission thread
edge_weights = defaultdict(int)
for submission_id, comments in submission_dict.items():
    authors = [comment['author'] for comment in comments if comment['author'] != 'deleted']
    for i in range(len(authors)):
        for j in range(i + 1, len(authors)):
            if authors[i] != authors[j]:
                edge_weights[(authors[i], authors[j])] += 1

# Add edges with weight
for (author1, author2), weight in edge_weights.items():
    G.add_edge(author1, author2, weight=weight)

# Filter to get around 50 nodes
nodes_to_keep = list(G.nodes)[:50]
H = G.subgraph(nodes_to_keep)

# Filter edges to reduce their number, keep the top weighted edges
edges = sorted(H.edges(data=True), key=lambda x: x[2]['weight'], reverse=True)
edges_to_keep = edges[:len(edges)//2]  # Keep only the top half edges

# Create a new graph with the filtered edges
H_filtered = nx.Graph()
H_filtered.add_nodes_from(H.nodes(data=True))
H_filtered.add_edges_from([(u, v) for u, v, w in edges_to_keep])

# Analyze the network
# Degree Centrality
degree_centrality = nx.degree_centrality(H_filtered)

# Betweenness Centrality
betweenness_centrality = nx.betweenness_centrality(H_filtered)

# Define the file paths
nodes_file_path = 'C:/Users/yativ/OneDrive/Desktop/nodes2.csv'
edges_file_path = 'C:/Users/yativ/OneDrive/Desktop/edges2.csv'

# Export Nodes
with open(nodes_file_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Id', 'Label', 'DegreeCentrality', 'BetweennessCentrality', 'Subreddit'])
    for node in H_filtered.nodes:
        # Find the subreddit for the node
        subreddits = set(entry['subreddit'] for entry in data if entry['author'] == node)
        subreddit_list = ', '.join(subreddits)
        writer.writerow([node, node, degree_centrality[node], betweenness_centrality[node], subreddit_list])

# Export Edges
with open(edges_file_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Source', 'Target'])
    for edge in H_filtered.edges:
        writer.writerow([edge[0], edge[1]])
