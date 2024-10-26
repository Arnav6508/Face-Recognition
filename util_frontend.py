import tkinter as tk
from tkinter import messagebox
from tkmacosx import Button

def get_button(window, text, color, command, fg='white'):
    button = Button(
                        window,
                        text=text,
                        activebackground="black",
                        activeforeground="white",
                        fg=fg,
                        bg=color,
                        command=command,
                        height=60,
                        width=200,
                        font=('Helvetica bold', 20)
                    )

    return button

def get_img_label(window):
    label = tk.Label(window)
    label.grid(row = 0, column = 0)
    return label

def get_text_label(window, text):
    label = tk.Label(window, text=text)
    label.config(font=("sans-serif", 21), justify="left")
    return label

def get_entry_text(window):
    txt = tk.Text(
                    window,
                    height=1,
                    width=15, 
                    font=("Arial", 32)
                )
    return txt

def msg_box(title, description):
    messagebox.showinfo(title, description)