import tkinter as tk
from PIL import Image, ImageTk
from detector import *

def video():
    ret, frame = cap.read()
    if ret:
        
        frame = cv2.flip(frame, 1)
        
        window_width = root.winfo_width()
        window_height = root.winfo_height()

        face = None
        val = 9
        while face is None and val >= 0:
            face, coords = get_face(frame, val, modelo1)
            val -= 1

        if face is not None and coords is not None:
            pred = predict(modelo2, face, emotions)
            show_yes(frame, coords, pred)

        frame = cv2.resize(frame, (window_width, window_height))

        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
        canvas.imgtk = imgtk

    root.after(10, video)
    
def end():
    cap.release() 
    cv2.destroyAllWindows() 
    root.destroy() 

root = tk.Tk()
root.title("Emotion Detector")
root.attributes('-fullscreen', True)

modelo1, modelo2, emotions = cargar()
cap = cv2.VideoCapture(0)

canvas = tk.Canvas(root)
canvas.pack(fill=tk.BOTH, expand=True)

close_button = tk.Button(root, text="X", command=end, bg="red",
                         fg="red", font=("Helvetica",18 , "bold"), borderwidth=0,
                         width=2, height=2)

close_button.place(x=0, y=0)

video()

root.mainloop()




        