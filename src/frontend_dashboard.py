import gradio as gr
from recommendor_logics import recommend_books, books

categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["All"] + ["Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

with gr.Blocks(theme= gr.themes.Glass()) as dashboard:
    gr.Markdown("# Semantic Book Recommender")

    with gr.Row():
        user_query = gr.Textbox(label="Please enter a description of a book you like",
                           placeholder="e.g.. A story about friendship and adventure")
        category_dropdown = gr.Dropdown(choices = categories, label = "Select a category", value = "All")
        tone_dropdown = gr.Dropdown(choices = tones, label = "Select an emotional tone", value = "All")
        submit_button = gr.Button("recommend books")
    
    gr.Markdown("## Recommendations")
    output = gr.Gallery(label="Recommended Books", columns = 8, rows = 2)

    submit_button.click(fn= recommend_books, 
                        inputs= [user_query, category_dropdown, tone_dropdown], 
                        outputs= output)
    
if __name__ == "__main__":
    dashboard.launch()