https://github.com/Hk669/discordAI-bot/assets/96101829/05911ed4-2e9f-4bc0-9ac3-37aac84fa34c

# Discord AI Bot

Discord AI Bot utilizes RAG (Retrieval-Augmented Generation) for generating responses to user queries or prompts.
It employs a LangChain, a custom language model chain, for generating contextually relevant responses.


## Features

- **AI Response Generation**: Utilizes RAG model from [rag.py](/rag.py) to generate AI responses based on user queries.
- **Command-based Interaction**: Supports commands such as `/ai` and `/bot` for interacting with the bot.

## Installation

To install and run the Discord Rag Bot, follow these steps:

1. Clone the repository:

    ```bash
    git clone https://github.com/Hk669/discordAI-bot.git
    ```

2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Set up environment variables:

    - Create a `.env` file in the project directory.
    - Add your Discord bot token in the `.env` file:

        ```env
        token=YOUR_DISCORD_BOT_TOKEN
        OPENAI_API_KEY = YOUR_API_KEY
        ```

4. Run the bot:

    ```bash
    python bot.py
    ```

## Usage

- **AI Response**: Use the `/ai` command followed by your query to get a response from the AI.
    ```
    /ai How does RAG model work?
    ```
    ```
    /gpt How does security in blockchain work?
    ```
    ```
    /bot How does Blockchain work?
    ```


## Contributing

Contributions are welcome! If you'd like to contribute to the project, feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


---
