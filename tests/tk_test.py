import tkinter as tk
from tkinter import messagebox

def say_hi():
    messagebox.showinfo("Tkinter test", "Hello world! Tkinter is working ✅")

root = tk.Tk()
root.title("Tkinter Hello World Test")
root.geometry("360x160")

label = tk.Label(root, text="Hello world!", font=("Arial", 18))
label.pack(pady=20)

btn = tk.Button(root, text="Click me", command=say_hi)
btn.pack()

root.mainloop()
