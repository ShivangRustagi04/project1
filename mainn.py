import streamlit as st
from transformers import pipeline

# Initialize the LLM pipeline
model_name = "gpt2"  # You can change this to another model if desired
llm_pipeline = pipeline("text-generation", model=model_name)

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

# Function to find top 100 books
def find_top_books(genre, num_books):
    prompt = f"List the top {num_books} books in the {genre} genre."
    response = llm_pipeline(prompt, max_length=1000, num_return_sequences=1)
    books = response[0]['generated_text'].strip().split('\n')
    top_books = [book.strip() for book in books if book.strip() and not book.strip().isdigit()][:num_books]
    result = f"Top {num_books} books in the {genre} genre:"
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

    # Define tasks
    task1 = Task("Find top 100 books in the fiction genre", find_top_books, "fiction", 100)
    agent.add_task(task1)

    # Execute the first task to get the top 100 books
    top_100_books, top_100_result = task1.execute()
    st.success(top_100_result)

    # Add the second task with the output of the first task
    task2 = Task("Find top 10 books from top 100", find_top_10_from_100, top_100_books)
    agent.add_task(task2)

    # Execute the second task to get the top 10 books
    top_10_books, top_10_result = task2.execute()
    st.success(top_10_result)

    # Add the third task with the output of the second task
    task3 = Task("Find 1 book from top 10 for user", find_1_from_top_10, top_10_books)
    agent.add_task(task3)

    # Execute the third task to get the selected book
    selected_book, selected_book_result = task3.execute()
    st.success(selected_book_result)

    # Add a final task to conclude the workflow with a thank-you message
    task4 = Task("Close the task and conclude workflow with a thank-you message", lambda: "Thank you for using the book finder agent!")
    agent.add_task(task4)

    # Execute the final task
    final_result = task4.execute()
    st.success(final_result)

if __name__ == "__main__":
    main()
