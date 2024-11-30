import logging
import pandas as pd
import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
from src.data_processing import load_data, preprocess_data
from src.collaborative import user_based_recommendations
from src.graph_analysis import build_graph, recommend_movies
import plotly.graph_objs as go

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
def display_recommendations(movie_ids, movies_df):
    """
    Convert movie IDs to movie titles and create interactive cards with IMDB links.
    """
    # Load the links file
    links_df = pd.read_csv('data/links.csv')
    
    # Get recommendations and merge with links
    recommendations = movies_df[movies_df['movieId'].isin(movie_ids)][['movieId', 'title']]
    recommendations = recommendations.merge(links_df[['movieId', 'imdbId']], on='movieId', how='left')
    
    # Format IMDB IDs with leading zeros
    recommendations['imdbId'] = recommendations['imdbId'].apply(lambda x: f"tt{str(x).zfill(7)}")
    # Create a grid layout for cards
    cols = st.columns(3)
    
    for idx, (_, movie) in enumerate(recommendations.iterrows()):
        with cols[idx % 3]:
            st.markdown(
                f"""
                <div style="
                    padding: 1rem;
                    border-radius: 10px;
                    border: 1px solid #ddd;
                    margin: 0.5rem 0;
                    background: white;
                    box-shadow: 2px 2px 5px rgba(0,0,0,0.1);">
                    <div style="
                        background: #f0f0f0;
                        height: 200px;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        margin-bottom: 1rem;">
                        🎬
                    </div>
                    <h4 style="color: black;">{movie['title']}</h4>
                    <a href="http://www.imdb.com/title/{movie['imdbId']}" 
                        target="_blank" 
                        style="
                        display: inline-block;
                        padding: 0.5rem 1rem;
                        background: #2e7d32;
                        color: white;
                        text-decoration: none;
                        border-radius: 5px;
                        margin-top: 0.5rem;">
                        View on IMDB
                    </a>
                </div>
                """,
                unsafe_allow_html=True
            )
    
    return recommendations['title'].tolist()


def create_interactive_graph_visualization(G, max_nodes=300, max_edges=200):
    try:
        # Limit graph size
        if len(G.nodes) > max_nodes:
            G = nx.Graph(G.subgraph(list(G.nodes)[:max_nodes]))
        if len(G.edges) > max_edges:
            G = nx.Graph(G.edge_subgraph(list(G.edges)[:max_edges]))

        # Calculate layout
        pos = nx.spring_layout(G)

        # Create traces for edges
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        )

        # Create traces for nodes
        node_x = []
        node_y = []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)

        node_adjacencies = [len(list(G.neighbors(node))) for node in G.nodes()]
        node_text = [f'Node {node}<br># of connections: {adj}' 
                    for node, adj in zip(G.nodes(), node_adjacencies)]

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers',
            hoverinfo='text',
            text=node_text,
            marker=dict(
                showscale=True,
                colorscale='YlGnBu',
                size=10,
                color=node_adjacencies,
                colorbar=dict(
                    thickness=15,
                    title='Node Connections',
                    xanchor='left',
                    titleside='right'
                )
            )
        )

        # Create figure
        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
        )

        st.plotly_chart(fig)

    except Exception as e:
        logging.error(f"Error in create_interactive_graph_visualization: {e}")
        st.error("An error occurred while creating the visualization.")

def main():
    # Page configuration and styling
    st.set_page_config(page_title="Movie Recommender", layout="wide", initial_sidebar_state="expanded")
    
    # Custom CSS for better styling
    st.markdown("""
        <style>
        .main {
            padding: 2rem;
        }
        .stButton>button {
            width: 100%;
        }
        .upload-header {
            font-weight: bold;
            margin-bottom: 1rem;
        }
        </style>
    """, unsafe_allow_html=True)

    # Animated title using HTML/CSS
    st.markdown("""
        <h1 style='text-align: center; color: #2e7d32; animation: fadeIn 2s;'>
            🎬 Movie Recommendation System
        </h1>
        <style>
        @keyframes fadeIn {
            0% { opacity: 0; }
            100% { opacity: 1; }
        }
        </style>
    """, unsafe_allow_html=True)

    # Sidebar for file uploads
    with st.sidebar:
        st.markdown("<h3 class='upload-header'>📂 Upload Your Data</h3>", unsafe_allow_html=True)
        movies_file = st.file_uploader("Upload movies CSV", type="csv")
        ratings_file = st.file_uploader("Upload ratings CSV", type="csv")

    if movies_file and ratings_file:
        with st.spinner('🔄 Loading data...'):
            movies_df = pd.read_csv(movies_file)
            ratings_df = pd.read_csv(ratings_file)
        st.success('✅ Data loaded successfully!')
        with st.spinner('⚙️ Preprocessing data...'):
            user_movie_matrix =  preprocess_data(ratings_df, movies_df)
        st.success('✅ Data preprocessing complete!')

        # Create tabs for different sections
        tab1, tab2 = st.tabs(["🎯 Get Recommendations", "📊 Visualization"])

        with tab1:


            # User input section with better styling
            st.markdown("### 👤 Select User")
            col1, col2, col3 = st.columns([1,2,1])
            with col2:
                user_id = st.number_input("Enter User ID", min_value=1, step=1)
                generate_button = st.button("Generate Recommendations", use_container_width=True)

            if generate_button and user_id:
                # User-based recommendations with progress
                with st.spinner('🤖 Generating user-based recommendations...'):
                    user_recommendations = user_based_recommendations(user_id, user_movie_matrix)
                    user_recommended_titles = display_recommendations(
                        movie_ids=[i for i, score in enumerate(user_recommendations) if score > 0],
                        movies_df=movies_df,
                    )
                
                # Display recommendations in an expander
                with st.expander("👥 User-Based Recommendations", expanded=True):
                    for i, title in enumerate(user_recommended_titles, 1):
                        st.write(f"{i}. {title}")

                # Graph-based recommendations
                with st.spinner('🕸️ Generating graph-based recommendations...'):
                    graph = build_graph(ratings_df)
                    graph_recommendations = recommend_movies(graph, user_id)
                    graph_recommended_titles = display_recommendations(graph_recommendations, movies_df)
                    print(graph_recommended_titles)
                with st.expander("🎯 Graph-Based Recommendations", expanded=True):
                    for i, title in enumerate(graph_recommended_titles, 1):
                        st.write(f"{i}. {title}")

        with tab2:
            if 'graph' in locals():
                st.markdown("### 🕸️ Interactive Network Visualization")
                with st.spinner('Creating visualization...'):
                    create_interactive_graph_visualization(graph)
    else:
        # Welcome message when no files are uploaded
        st.markdown("""
            <div style='text-align: center; padding: 2rem;'>
                <h3>👋 Welcome to the Movie Recommender!</h3>
                <p>Please upload the required CSV files in the sidebar to get started.</p>
            </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
