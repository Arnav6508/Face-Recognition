import tkinter as tk
import utils_frontend
import cv2 
import datetime
import numpy as np
from PIL import Image, ImageTk
from utils import add_new_embedding_from_image
from test import test_from_image

class App():
    def __init__(self):
        # open main window
        self.main_window = tk.Tk()
        self.main_window.geometry("1050x520+350+100")

        # login button
        self.login_button = utils_frontend.get_button(self.main_window, 'Login', 'green', self.login)
        self.login_button.place(x = 830, y = 320)

        # register button
        self.register_button = utils_frontend.get_button(self.main_window, 'Register', 'gray', self.register, fg = 'black')
        self.register_button.place(x = 830, y = 400)

        # label for webcam
        self.webcam_label = utils_frontend.get_img_label(self.main_window)
        self.webcam_label.place(x = 10, y = 0, width = 800, height = 500)
        self.add_webcam(self.webcam_label)

        # db and log file path
        self.db_dir = './test.db'
        self.log_path = './log.txt'

    def add_webcam(self, webcam_label):
        if 'cap' not in self.__dict__:
            self.cap = cv2.VideoCapture(0)
        
        self.label = webcam_label
        self.process_webcam()
    
    def process_webcam(self):
        ret, frame = self.cap.read()

        self.image_cv2 = frame
        img_ = cv2.cvtColor(self.image_cv2, cv2.COLOR_BGR2RGB)
        self.image_pil = Image.fromarray(img_).resize((800,500), Image.Resampling.LANCZOS)

        imgtk = ImageTk.PhotoImage(image = self.image_pil)
        self.label.imgtk = imgtk
        self.label.configure(image = imgtk)

        self.label.after(20, self.process_webcam)


    def start(self):
        self.main_window.mainloop()

    def login(self): 
        self.name = test_from_image(self.image_cv2, self.db_dir)
        if self.name == None: utils_frontend.msg_box('Failure!','Not recognised, \nPlease register')
        else : 
            utils_frontend.msg_box('Success!',f'Welcome, {self.name}! \n Attendance Marked')
            with open(self.log_path, 'a') as f:
                f.write(f'{self.name}, {datetime.datetime.now()}')
                f.close()

    def register(self): 
        # open new window
        self.register_window = tk.Toplevel(self.main_window)
        self.register_window.geometry("1050x520+370+120")

        # text label
        self.text_label_for_register = utils_frontend.get_text_label(self.register_window, "Enter Username:")
        self.text_label_for_register.place(x = 750, y = 100)

        # text box for inputting name
        self.text_for_register = utils_frontend.get_entry_text(self.register_window)
        self.text_for_register.place(x = 750, y = 150)

        # accept button
        self.accept_button = utils_frontend.get_button(self.register_window, 'Accept', 'green', self.accept)
        self.accept_button.place(x = 750, y = 320)

        # try again button
        self.try_again_button = utils_frontend.get_button(self.register_window, 'Try Again', 'red', self.try_again)
        self.try_again_button.place(x = 750, y = 400)

        # label for photo capture
        self.capture_label = utils_frontend.get_img_label(self.register_window)
        self.capture_label.place(x = 10, y = 0, width = 700, height = 500)
        self.add_img_to_label(self.capture_label)


    def add_img_to_label(self, label):
        # Take most recent photo captured by webcam
        self.register_imgtk = ImageTk.PhotoImage(image = self.image_pil)
        label.imgtk = self.register_imgtk
        label.configure(image = self.register_imgtk)

    def accept(self):
        self.register_name = self.text_for_register.get("1.0", "end-1c")
        img_pil = ImageTk.getimage(self.register_imgtk)
        img_cv = np.array(img_pil)
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
        add_new_embedding_from_image(img_cv, self.register_name, self.db_dir)

        utils_frontend.msg_box('Success!','User registered successsfully!')
        self.register_window.destroy()

    def try_again(self):
        # go back to main window
        self.register_window.destroy()


if __name__ == '__main__':
    app = App()
    app.start()