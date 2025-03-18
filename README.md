# LLM Semantic Book Recommender 📚

This is a Semantic Book Recommender that uses Large Language Model to recommend books based on user input. The system takes a user-provided book description, filters recommendations based on category and emotional tone, and returns the top-matching books with their covers and descriptions.

--- 

## 🚀 Features
✅ **Semantic Search:** Recommends books based on the semantic meaning of the user input.  
✅ **Category Filtering:** Allows filtering by book category (e.g., Fiction, Thriller).  
✅ **Emotional Tone Sorting:** Filters books based on emotional tone like *Happy, Surprising, Angry, Suspenseful, and Sad*.  
✅ **Interactive Dashboard:** Built with **Gradio** for an easy-to-use and responsive UI.  
✅ **AI-Powered Embeddings:** Uses **Google Generative AI Embeddings** for accurate semantic search.  

---

There are five main components, from data preprocessing to final application development
- Text data cleaning (code in the notebook data-exploration.ipynb)
- Semantic (vector) search and how to build a vector database (code in the notebook vector_search.ipynb). This allows users to find the most similar books to a natural language query (e.g., "a book about a person seeking revenge").
- Doing text classification using zero-shot classification in LLMs (code in the notebook text_classification.ipynb). This allows us to classify the books as "fiction" or "non-fiction", creating a facet that users can filter the books on.
- Doing sentiment analysis using LLMs and extracting the emotions from text (code in the notebook book_sentiment_analysis.ipynb). This will allow users to sort books by their tone, such as how suspenseful, joyful or sad the books are.
- Creating a web application using Gradio for users to get book recommendations (code in the file src/frontend-dashboard.py).

---

A requirements.txt file contains all the project dependencies to run the application.


![Book recommender UI](evidence/Screenshot_UI_1.png)

