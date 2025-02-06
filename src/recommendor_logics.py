import numpy as np
import pandas as pd

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

from dotenv import load_dotenv
load_dotenv()

books = pd.read_csv("../notebooks/books_emotions_data.csv")
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books['large_thumbnail'] = np.where(
    books['large_thumbnail'].isna(), 
    "no_book_cover.jpg",
    books['large_thumbnail']
) 

raw_documents = TextLoader('../notebooks/books_tagged_description.txt', encoding='utf-8').load()
text_splitter = CharacterTextSplitter(chunk_size=0, chunk_overlap=0, separator='\n')
documents = text_splitter.split_documents(raw_documents)
#embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
db_books = Chroma.from_documents(documents, embedding=embeddings)

def retrieve_semantic_recommendations(
        query: str,
        category: str = None,
        tone: str = None,
        initial_top_k : int = 50,
        final_top_k : int = 16
)->pd.DataFrame : 
    """
     Retrieves semantic book recommendations with optional filtering and sorting.
    Args:
        query: Search query
        category: Filter by category (default: None, "All" to bypass)
        tone: Sort by emotional tone (default: None, options: Happy, Surprising, Angry, Suspenseful, Sad)
        initial_top_k: Initial result set size (default: 50)
        final_top_k: Final result set size (default: 16)
    Returns:
        pd.DataFrame: Top-matching book recommendations with metadata and emotional tone scores.
    """
    recs = db_books.similarity_search(query, k=initial_top_k)
    books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
    book_recs = books[books["isbn13"].isin(books_list)].head(initial_top_k)

    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"]==category].head(final_top_k)
    else:
        book_recs = book_recs.head(final_top_k)

    if tone == "Happy":
        book_recs.sort_values(by="joy", ascending=False, inplace=True)
    elif tone == "Surprising":
        book_recs.sort_values(by="surprise", ascending=False, inplace=True)
    elif tone == "Angry":
        book_recs.sort_values(by="anger", ascending=False, inplace=True)
    elif tone == "Suspenseful":
        book_recs.sort_values(by="fear", ascending=False, inplace=True)
    elif tone == "Sad":
        book_recs.sort_values(by="sadness", ascending=False, inplace=True)
        
    return book_recs

def recommend_books(
        query: str,
        category: str,
        tone: str
):
    """
    Returns book recommendations based on query, category, and tone.
    Args:
        query: Search query
        category: Book category
        tone: Desired book tone
    Returns:
        List of tuple containing : thumbnail URL & formatted book caption.
    """
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []
    for _, row in recommendations.iterrows():
        description = row["description"]
        truncated_desc_split = description.split()
        truncated_description = " ".join(truncated_desc_split[:30]) + "..."

        authors_split = row["authors"].split(";")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])} and {authors_split[-1]}"
        else:
            authors_str = row["authors"]

        caption = f"{row['title']} by {authors_str} : {truncated_description}"
        results.append((row["large_thumbnail"], caption))
    return results