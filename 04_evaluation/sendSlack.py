#import json
#import requests
#import os
from slackclient import SlackClient

class SendSlack():
    sc = SlackClient("xoxb-478367944196-601656076022-wUFTtb3cLMevFMTNmrK1xXOw")

    @staticmethod
    def sendFile(filePath, fileTitle):
        with open(filePath, 'rb') as fileContent:
            SendSlack.sc.api_call(
                "files.upload",
                channels='the-neural-net',
                file=fileContent,
                title=fileTitle,
            )
    
    @staticmethod
    def sendText(text_content):
        SendSlack.sc.api_call(
            "chat.postMessage",
            channel="the-neural-net",
            text=text_content
        )
           
# SendSlack.sendText("Hello my friends....im a python code")
            
# with open('bild.png', 'rb') as file_content:
#     SendSlack.sendFile(file_content, "test")