# Softnerve_Technology
# LLM-based Book Finder Agent

## Overview

This project implements a simple autonomous agent to help users find top books in any genre using a Hugging Face model and reinforcement learning techniques.

## Workflow

1. **User Input**: The user specifies a genre.
2. **Fetch Top 100 Books**: The agent fetches the top 100 books in the specified genre.
3. **Select Top 10 Books**: The agent selects the top 10 books from the list of 100.
4. **Find One Book**: The agent helps the user select one book based on user preferences.
5. **Conclude Task**: The agent concludes the task with a thank-you message.

## Tools Used

- **FastAPI**: For creating a REST API.
- **Hugging Face Transformers**: For generating book recommendations.
- **Python**: Main programming language.

## Reasoning

- **FastAPI**: Chosen for its simplicity and speed in creating REST APIs.
- **Hugging Face Transformers**: Chosen for their powerful natural language processing capabilities to generate book recommendations based on user input.

## Future Enhancements

- Improve error handling and robustness.
- Enhance the user interface for better user experience.
- Add more sophisticated filtering and recommendation logic.
