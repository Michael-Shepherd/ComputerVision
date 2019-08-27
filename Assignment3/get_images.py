from tkinter import *
from PIL import Image, ImageTk
import cv2
import os


if __name__ == "__main__":
    root = Tk()
    points = []
    # setting up a tkinter canvas with scrollbars
    frame = Frame(root, bd=2, relief=SUNKEN)
    frame.grid_rowconfigure(0, weight=1)
    frame.grid_columnconfigure(0, weight=1)
    xscroll = Scrollbar(frame, orient=HORIZONTAL)
    xscroll.grid(row=1, column=0, sticky=E + W)
    yscroll = Scrollbar(frame)
    yscroll.grid(row=0, column=1, sticky=N + S)
    canvas = Canvas(frame, bd=0)
    canvas.grid(row=0, column=0, sticky=N + S + E + W)
    xscroll.config(command=canvas.xview)
    yscroll.config(command=canvas.yview)
    frame.pack(fill=BOTH, expand=1)

    # adding the image
    im_temp = Image.open("./CVassignment3_files/lego1.jpg")
    img = im_temp.resize((640, 480), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    canvas.create_image(0, 0, image=img, anchor="nw")
    canvas.config(scrollregion=canvas.bbox(ALL))

    # function to be called when mouse is clicked

    def printcoords(event):
        # outputting x and y coords to console
        print(event.x, event.y)
        points.append([event.x, event.y])

    # mouseclick event
    canvas.bind("<Button 1>", printcoords)

    root.mainloop()
    for p in points:
        print(f"[{p[0]},{p[1]}]")


