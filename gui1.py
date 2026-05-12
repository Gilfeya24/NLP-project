import tkinter as tk
from tkinter import scrolledtext,ttk,messagebox
import os
import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import load_model
import webbrowser
import datetime
import sys

#defining the chatbot gui class
class ChatbotGUI:
    def __init__(self, master):
        self.master = master
        self.setup_gui()
        self.load_chatbot_model()
        self.conversation_history = []
#setting up the gui
    def setup_gui(self):
        self.master.title("Chatbot")
        self.master.geometry("400x500")
        self.master.configure(bg="#f0f0f0") #background colour

        style=ttk.Style()
        style.theme_use('clam')

        main_frame=ttk.Frame(self.master,padding="10")
        main_frame.pack(fill=tk.BOTH,expand=True)

        self.chat_history=scrolledtext.ScrolledText(main_frame,state='disabled',wrap=tk.WORD,width=50,height=20,font=('Arial',10))
        self.chat_history.pack(padx=10,pady=10,fill=tk.BOTH,expand=True)
        self.chat_history.configure(state=tk.DISABLED)

        input_frame=ttk.Frame(main_frame)
        input_frame.pack(fill=tk.X,pady=5)

        self.user_input=tk.Entry(input_frame,width=50,font=('Arial',10))
        self.user_input.pack(side=tk.LEFT,padx=(0,5),fill=tk.X,expand=True)
        self.user_input.bind("<Return>",lambda event: self.send_message())

        self.send_button=ttk.Button(input_frame,text="Send",command=self.send_message)
        self.send_button.pack(side=tk.RIGHT)

        button_frame=ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X,pady=5)

        self.clear_button=ttk.Button(button_frame,text="Clear Chat",command=self.clear_chat)
        self.clear_button.pack(side=tk.LEFT,padx=(0,5))
        self.save_button=ttk.Button(button_frame,text="Save Chat",command=self.save_chat)
        self.save_button.pack(side=tk.LEFT)

        self.help_button=ttk.Button(button_frame,text="Help",command=self.show_help)
        self.help_button.pack(side=tk.RIGHT)

    def load_chatbot_model(self):
        self.lemmatizer = WordNetLemmatizer()
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, '..'))
        intents_path = os.path.join(script_dir, 'intents.json')
        model_path = os.path.join(project_root, 'chatbot_model.h5')
        words_path = os.path.join(project_root, 'words.pkl')
        classes_path = os.path.join(project_root, 'classes.pkl')

        self.intents = json.loads(open(intents_path, 'r', encoding='utf-8').read())
        self.model = load_model(model_path)
        self.words = pickle.load(open(words_path, 'rb'))
        self.classes = pickle.load(open(classes_path, 'rb'))

        for resource in ['punkt', 'wordnet', 'omw-1.4']:
            try:
                if resource == 'punkt':
                    nltk.data.find('tokenizers/punkt')
                else:
                    nltk.data.find(f'corpora/{resource}')
            except LookupError:
                nltk.download(resource, quiet=True)

    def send_message(self):
        user_message = self.user_input.get().strip()
        self.user_input.delete(0, tk.END)
        if user_message:
            self.update_chat_history("You: " + user_message)
            try:
                bot_response = self.get_bot_response(user_message)
            except Exception as e:
                bot_response = f"Sorry, I encountered an error: {e}"
                print(f"Chatbot error: {e}", file=sys.stderr)
            self.update_chat_history("Bot: " + bot_response)
            self.conversation_history.append((user_message, bot_response))

    def update_chat_history(self, message):
        self.chat_history.configure(state=tk.NORMAL)
        self.chat_history.insert(tk.END, message + "\n")
        self.chat_history.configure(state=tk.DISABLED)
        self.chat_history.see(tk.END)
    
    def get_bot_response(self, user_message):
        if user_message.lower() in ["exit","quit","bye"]:
            return 'goodbye'
        elif user_message.lower().startswith("search "):
            query=user_message[7:]
            webbrowser.open(f"https://www.google.com/search?q={query}")
            return f"Searching for {query} on Google..."
        elif user_message.lower()=="time":
            return(
                f"the current time is {datetime.datetime.now().strftime('%H:%M:%S')}."


            )
        else:
            ints=self.predict_class(user_message)
            res=self.get_response(ints,self.intents)
            return res

        '''cleaning up sentence and bag of words'''
    def clean_up_sentence(self,sentence):
        return [
            self.lemmatizer.lemmatize(word.lower()) for word in nltk.word_tokenize(sentence)
        ]
    
    def bag_of_words(self,sentence):
        sentence_words=self.clean_up_sentence(sentence)
        bag=[1 if word in sentence_words else 0 for word in self.words]
        return np.array(bag)

    '''predicting class and getting response''' 
    def predict_class(self,sentence):
        bow=self.bag_of_words(sentence)
        res=self.model.predict(np.array([bow]))[0]
        ERROR_THRESHOLD=0.25
        results=[[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
        results.sort(key=lambda x:x[1],reverse=True)
        return_list=[]
        for r in results:
            return_list.append({"intent":self.classes[r[0]],"probability":str(r[1])})
        return return_list

    def get_response(self,intents_list,intents_json):
        if not intents_list:
            return ' i am not sure how to respond to that.'
        tag=intents_list[0]['intent']
        for intent in self.intents["intents"]:
            if intent['tag']==tag:
                return random.choice(intent['responses'])
        return "i am sorry i don't have a specific response for that"

    def clear_chat(self):
        self.chat_history.configure(state=tk.NORMAL)
        self.chat_history.delete(1.0, tk.END)
        self.chat_history.configure(state=tk.DISABLED)
        self.conversation_history.clear()

    def save_chat(self):
        filename=f"chat_history_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"   
        with open(filename,'w') as file:
            for user_msg,bot_msg in self.conversation_history:
                file.write(f"You: {user_msg}\n")
                file.write(f"Bot: {bot_msg}\n")
        messagebox.showinfo("Chat Saved", f"Chat history saved as {filename}")
    
    def show_help(self):
        help_text=(
            "Welcome to the Chatbot!\n\n"
            "You can ask me anything or use the following commands:\n"
            "- Type 'time' to get the current time.\n"
            "- Type 'search <query>' to search Google for a specific query.\n"
            "- Type 'exit', 'quit', or 'bye' to end the conversation.\n\n"
            "Feel free to chat with me!"
        )
        messagebox.showinfo("Help", help_text)
    
if __name__ == "__main__":
    root=tk.Tk()
    chatbot_gui=ChatbotGUI(root)
    root.mainloop()

    




    

