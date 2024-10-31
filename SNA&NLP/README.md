
# Kendrick vs. Drake: Natural Language Processing & Social Network Analysis Project

## Overview
This project explores the debate between Kendrick Lamar and Drake using Natural Language Processing (NLP) and Social Network Analysis (SNA). The focus is on analyzing Reddit discussions, building a social graph, and examining key metrics to understand user behavior and sentiment trends about the artists.

## Features
- **Data Collection**: Gathered data from `r/kendricklamar` and `r/Drizzy` subreddits using the PRAW library.
- **Graph Construction**: Created a network graph of Reddit interactions between users discussing the artists.
- **Analysis Metrics**:
  - **Centrality Measures**: Degree and betweenness centrality to find influential users.
  - **Community Detection**: Identified clusters within the graph.
- **Text Processing**:
  - Applied tokenization, stop word removal, and lemmatization for clean text analysis.
- **Sentiment Analysis**: Assessed sentiment polarity of posts.
- **Topic Modeling**: Used Latent Dirichlet Allocation (LDA) to identify main topics.
- **Visualization**:
  - Generated word clouds for mentions of both artists.
  - Created sentiment distribution histograms.

## Files
- **NLP&SNA.py**: Contains functions for data collection, graph construction, and metric calculations.
- **Report (PDF)**: Includes detailed analysis findings and visualizations.

## Technologies Used
- **Python Libraries**:
  - `praw` for data extraction from Reddit
  - `networkx` for graph building and analysis
  - `TextBlob` and `NLTK` for text processing and sentiment analysis
  - `Gensim` for topic modeling
  - `matplotlib` and `wordcloud` for visualization

## Usage
1. **Data Collection**:
   Configure PRAW with your Reddit credentials:
   ```python
   reddit = praw.Reddit(
       client_id="YOUR_CLIENT_ID",
       client_secret="YOUR_CLIENT_SECRET",
       user_agent="YOUR_USER_AGENT",
       username="YOUR_USERNAME",
       password="YOUR_PASSWORD"
   )
   ```

2. **Run Analysis**:
   Run `NLP&SNA.py` to collect data, build the graph, and export results to CSV files.

3. **Graph Visualization**:
   Load the CSV files into Gephi or another tool for visualization.

## Results Summary
Key insights include:
- **Community Structure**: Central users and interactivity patterns in the network.
- **Sentiment Analysis**: Comparative sentiment scores for Kendrick and Drake.
- **Topic Modeling**: Identified prominent themes from discussions.


