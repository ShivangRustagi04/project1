
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
import torch
import requests
from bs4 import BeautifulSoup

# Initialize the tokenizer and model
model_name = "EleutherAI/gpt-neo-2.7B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

# Apply LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none"
)
model = get_peft_model(model, lora_config)

class WorkflowAgent:
    def __init__(self):
        self.tasks = []

    def add_task(self, task):
        self.tasks.append(task)

    def execute_tasks(self):
        for task in self.tasks:
            result = task.execute()
            if result:
                st.success(result)

class Task:
    def __init__(self, name, function=None, *args, **kwargs):
        self.name = name
        self.function = function
        self.args = args
        self.kwargs = kwargs

    def execute(self):
        if self.function:
            return self.function(*self.args, **self.kwargs)
        return None

# Function to generate text using the LoRA model
def generate_text(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_length=1000, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

# Function to find top 100 books using the LLM
def find_top_books_llm(genre, num_books):
    prompt = f"List the top {num_books} books in the {genre} genre."
    response = generate_text(prompt)
    books = response.strip().split('\n')
    top_books = [book.strip() for book in books if book.strip() and not book.strip().isdigit()][:num_books]
    result = f"Top {num_books} books in the {genre} genre (LLM):"
    for i, book in enumerate(top_books):
        result += f"\n{i+1}. {book}"
    return top_books, result

# Function to find top 100 books using web scraping
def find_top_books_web(genre, num_books):
    url = f"https://www.goodreads.com/shelf/show/{genre}"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    books = soup.select("a.bookTitle span")
    top_books = [book.get_text() for book in books][:num_books]
    result = f"Top {num_books} books in the {genre} genre (Web Scraping):"
    for i, book in enumerate(top_books):
        result += f"\n{i+1}. {book}"
    return top_books, result

# Function to find top 10 books from top 100
def find_top_10_from_100(top_100_books):
    top_10_books = top_100_books[:10]  # Simulating the selection of top 10 from 100
    result = "Top 10 books from the top 100:"
    for i, book in enumerate(top_10_books):
        result += f"\n{i+1}. {book}"
    return top_10_books, result

# Function to find 1 book from top 10
def find_1_from_top_10(top_10_books):
    selected_book = top_10_books[0]  # Simulating the selection of 1 book from top 10
    result = f"Selected book from the top 10: {selected_book}"
    return selected_book, result

# Main Streamlit app
def main():
    st.title("Workflow Agent Demo")

    # Initialize the workflow agent
    agent = WorkflowAgent()

    genre = st.text_input("Enter book genre:", "fiction")

    # Define tasks
    task1_llm = Task("Find top 100 books in the genre using LLM", find_top_books_llm, genre, 100)
    task1_web = Task("Find top 100 books in the genre using Web Scraping", find_top_books_web, genre, 100)
    agent.add_task(task1_llm)
    agent.add_task(task1_web)

    # Execute the first task to get the top 100 books
    if "top_100_books_llm" not in st.session_state:
        st.session_state.top_100_books_llm, top_100_result_llm = task1_llm.execute()
        st.session_state.top_100_books_llm_result = top_100_result_llm
    st.success(st.session_state.top_100_books_llm_result)

    if "top_100_books_web" not in st.session_state:
        st.session_state.top_100_books_web, top_100_result_web = task1_web.execute()
        st.session_state.top_100_books_web_result = top_100_result_web
    st.success(st.session_state.top_100_books_web_result)

    # Compare results from LLM and Web Scraping
    st.write("Comparison of Top 100 Books (LLM vs Web Scraping):")
    comparison_result = "Books in both lists:\n"
    common_books = set(st.session_state.top_100_books_llm) & set(st.session_state.top_100_books_web)
    for book in common_books:
        comparison_result += f"- {book}\n"
    st.write(comparison_result)

    # Add the second task with the output of the first task
    task2 = Task("Find top 10 books from top 100", find_top_10_from_100, st.session_state.top_100_books_llm)
    agent.add_task(task2)

    # Execute the second task to get the top 10 books
    if "top_10_books" not in st.session_state:
        st.session_state.top_10_books, top_10_result = task2.execute()
        st.session_state.top_10_books_result = top_10_result
    st.success(st.session_state.top_10_books_result)

    # Add the third task with the output of the second task
    task3 = Task("Find 1 book from top 10 for user", find_1_from_top_10, st.session_state.top_10_books)
    agent.add_task(task3)

    # Execute the third task to get the selected book
    if "selected_book" not in st.session_state:
        st.session_state.selected_book, selected_book_result = task3.execute()
        st.session_state.selected_book_result = selected_book_result
    st.success(st.session_state.selected_book_result)

    st.success("Thank you for using the Workflow Agent!")

if __name__ == "__main__":
    main()
