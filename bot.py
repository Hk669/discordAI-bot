import discord
import os
# from gpt import gpt_response
from rag import genchain, retreival_chain
from dotenv import load_dotenv
load_dotenv()

discord_token = os.getenv('token')

class MyClient(discord.Client):
    async def on_request(self):
        print('Successfully loggedin', self.user)

    async def on_message(self, message):
        print(message.content)
        if message.author == self.user:
            return
        
        command, user_message = None, None
        for text in ['/ai', '/bot', '/gpt']:
            if message.content.startswith(text):
                command = message.content.split(' ')[0]
                user_message = message.content.replace(text, '')
                print(command, user_message)

        if command == '/ai' or command == '/bot' or command == '/gpt':
            chain = genchain()
            bot_response = retreival_chain(chain,query=user_message)
            await message.channel.send(f"<@{message.author}> here is your answer: {bot_response}")


intents = discord.Intents.default()
intents.message_content = True

client = MyClient(intents=intents)

if __name__ == '__main__':
    client.run(discord_token)