import streamlit as st
from transformers import pipeline

model_name = "gpt2"
llm_pipeline = pipeline("text-generation", model=model_name)

def find_top_books(genre, num_books):
    prompt = f"List the top {num_books} books in the {genre} genre."
    response = llm_pipeline(prompt, max_length=500, num_return_sequences=1)
    books = response[0]['generated_text'].strip().split('\n')
    top_books = [book.strip() for book in books if book.strip()][:num_books]
    return top_books

def main():
    st.title("LLM-based Book Finder Agent")

    st.header("Find Top Books")
    genre = st.text_input("Enter a genre:")
    num_books = st.slider("Select the number of books to find:", min_value=1, max_value=100, value=10, step=1)
    
    if st.button("Find Top Books"):
        if genre:
            top_books = find_top_books(genre, num_books)
            st.success(f"Top {num_books} books in the {genre} genre:")
            for i, book in enumerate(top_books):
                st.write(f"{i+1}. {book}")
        else:
            st.warning("Please enter a genre.")

    if st.button("Close Task and Conclude Workflow"):
        st.success("Thank you for using the book finder agent!")
        st.stop()

    st.sidebar.title("About")
    st.sidebar.info(
        "This is a simple LLM-based autonomous agent that finds top books in any genre."
        " Enter a genre and select the number of books to find to get started."
    )

if __name__ == "__main__":
    main()
