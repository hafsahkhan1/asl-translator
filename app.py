import cv2
import tkinter as tk
from PIL import Image, ImageTk

cap = cv2.VideoCapture(0)

def show_frame():
    button.pack_forget()
    _, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame)
    imgtk = ImageTk.PhotoImage(image=img)
    label.imgtk = imgtk
    label.configure(image=imgtk)
    label.after(10, show_frame)

window = tk.Tk()
window.geometry("550x550")
window.config(background="#F1F7ED")
window.title("ASL Translator")
label = tk.Label(window)
label.pack()
button = tk.Button(window, text="Begin Translator", bg="#91C7B1", command=show_frame)
button.pack()
window.mainloop()
cap.release()
cv2.destroyAllWindows()   