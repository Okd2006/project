from tkinter import *
import PIL
import numpy as np
import main
import pickle
import PIL.ImageDraw

BLACK=(0,0,0)

class PaintGUI:
    def __init__(self):
        self.root=Tk()
        self.root.title('Draw to Predict')

        self.brush_width=10
        self.current_color='#00ffff'

        self.cnv=Canvas(self.root, width=280,height=280,bg='black')
        self.cnv.pack()
        self.cnv.bind("<B1-Motion>",self.paint)

        self.image=PIL.Image.new("RGB",(280,280),BLACK)
        self.draw=PIL.ImageDraw.Draw(self.image)

        self.btn_frame=Frame(self.root)
        self.btn_frame.pack(fill=X)

        self.btn_frame.columnconfigure(0,weight=1)
        self.btn_frame.columnconfigure(1,weight=1)

        self.clear_btn=Button(self.btn_frame,text='Clear',command=self.clear)
        self.clear_btn.grid(row=0,column=0,sticky=W+E)

        self.predict_btn=Button(self.btn_frame,text='Predict',command=self.predict)
        self.predict_btn.grid(row=0,column=1,sticky=W+E)

        self.root.mainloop()

    def paint(self,event):
        x1,y1=event.x-1,event.y-1
        x2,y2=event.x+1,event.y+1
        self.cnv.create_rectangle(x1,y1,x2,y2,outline=self.current_color,fill=self.current_color,width=self.brush_width)
        self.draw.rectangle([x1,y1,x2+self.brush_width,y2+self.brush_width],outline=self.current_color,fill=self.current_color,width=self.brush_width)

    def clear(self):
        self.cnv.delete('all')
        self.draw.rectangle([0,0,1000,1000],fill='black')

    def predict(self):
        new_image = self.image.resize((28, 28))
        pix=list(new_image.getdata())
        l=[]
        for i in pix:
            l+=[i[0]]
        l=[l]
        l=np.array(l).T/255.
        with open('data.bin','rb') as f:
            W1=pickle.load(f)
            b1=pickle.load(f)
            W2=pickle.load(f)
            b2=pickle.load(f)

        main.test(W1,b1,W2,b2,l)


PaintGUI()