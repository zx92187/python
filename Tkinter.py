# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 20:00:24 2018

@author: mzhen
"""

#tkinter part 1 and 2
#https://www.youtube.com/watch?v=eXOQwzHsyqU&list=PLXO45tsB95cJU56K4EtkG0YNGBZCuDwAH
#https://www.youtube.com/watch?v=vMAK1oJAtkQ&list=PLXO45tsB95cJU56K4EtkG0YNGBZCuDwAH&index=2

import tkinter as tk

#set window and window name
window = tk.Tk()
window.title('my window')

#window size
window.geometry('600x300')

#label
var=tk.StringVar()
l =tk.Label(window,textvariable=var
            ,bg='green',font=('Arial',12)
            ,width=15,height=2)
#place the window
l.pack()
#define button that is not clicking
on_hit=False

def hit_me():
    global on_hit
    if on_hit==False:
        on_hit=True
        var.set('you hit me')
    else:
        on_hit=False
        var.set('')
                
#add button
b=tk.Button(window,text='hit me',width=15,height=2
            ,command=hit_me)
b.pack()
#run the program,mainloop means constant looping, that's why the window can 
#execute and refresh the command
window.mainloop()



#tkinter part 3
#https://www.youtube.com/watch?v=lVcM_V3KqOE&index=3&list=PLXO45tsB95cJU56K4EtkG0YNGBZCuDwAH

import tkinter as tk

#set window and window name
window = tk.Tk()
window.title('my window')

#window size
window.geometry('200x200')


#entry, show='*' will put * to the entered the characters
#show = 'None' will reveal the actual characters
e=tk.Entry(window,show='*')
e.pack()

#define functions
#insert_point will insert text to where the curor is pointed at
#insert_end will only insert text to the end
def insert_point():
    var=e.get()
    t.insert('insert',var)
    
def insert_end():
    var=e.get()
    t.insert('end',var)
#t.insert(2.2,var) means insert the text to the 3rd(starts from 0) row and 3rd column
    #t.insert(2.2,var)

#button
b1=tk.Button(window,text='insert point',width=15
             ,height=2,command=insert_point)
b1.pack()

b2=tk.Button(window,text='insert end',width=15
             ,command=insert_end)
b2.pack()

t=tk.Text(window,height=2)
t.pack()

#run the program,mainloop means constant looping, that's why the window can 
#execute and refresh the command
window.mainloop()
