import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import cv2
import imutils
from scipy.spatial import distance
from imutils import face_utils
from pygame import mixer
import dlib


mixer.init()
mixer.music.load("music.wav")


def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


thresh = 0.25
frame_check = 20
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor(r"C:\Users\kirti\Downloads\shape_predictor_68_face_landmarks (1).dat")
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
flag = 0
cap = None
monitoring = False


registered_users = {}

class Application(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        self.geometry("600x400")
        self.title("Driver Drowsiness Monitoring System")
        
        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}
        for F in (WelcomePage, LoginPage, RegisterPage, Dashboard, MonitoringPage):
            frame = F(container, self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")
        
        self.show_frame(WelcomePage)

    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()

class WelcomePage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Welcome to Driver Drowsiness Monitoring System", font=("Arial", 16))
        label.pack(pady=50)
        btn_login = tk.Button(self, text="Proceed to Login", command=lambda: controller.show_frame(LoginPage))
        btn_login.pack()

class LoginPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Login Page", font=("Arial", 16))
        label.pack(pady=20)
        
        self.username = tk.Entry(self)
        self.password = tk.Entry(self, show='*')
        self.username.pack(pady=5)
        self.password.pack(pady=5)

        btn_login = tk.Button(self, text="Login", command=lambda: self.login(controller))
        btn_login.pack(pady=5)

        btn_register = tk.Button(self, text="Register", command=lambda: controller.show_frame(RegisterPage))
        btn_register.pack(pady=5)

    def login(self, controller):
        
        username = self.username.get()
        password = self.password.get()
        if username in registered_users and registered_users[username] == password:
            controller.show_frame(Dashboard)
        else:
            messagebox.showerror("Login Failed", "Incorrect Username/Password")

class RegisterPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Register Page", font=("Arial", 16))
        label.pack(pady=20)

        self.new_username = tk.Entry(self)
        self.new_password = tk.Entry(self, show='*')
        self.new_username.pack(pady=5)
        self.new_password.pack(pady=5)

        btn_register = tk.Button(self, text="Register", command=self.register)
        btn_register.pack(pady=5)

        btn_back = tk.Button(self, text="Back to Login", command=lambda: controller.show_frame(LoginPage))
        btn_back.pack(pady=5)

    def register(self):
        username = self.new_username.get()
        password = self.new_password.get()

        if username and password:
            if username not in registered_users:
                registered_users[username] = password
                messagebox.showinfo("Registration Success", "User registered successfully!")
            else:
                messagebox.showerror("Registration Failed", "Username already exists!")
        else:
            messagebox.showerror("Registration Failed", "Please enter valid username and password.")

class Dashboard(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Dashboard", font=("Arial", 16))
        label.pack(pady=20)

        btn_home = tk.Button(self, text="Home", command=lambda: controller.show_frame(Dashboard))
        btn_home.pack(side=tk.LEFT, padx=5)
        
        btn_about = tk.Button(self, text="About", command=lambda: messagebox.showinfo("About", "This is a Driver Drowsiness Monitoring System."))
        btn_about.pack(side=tk.LEFT, padx=5)

        btn_start_monitoring = tk.Button(self, text="Start Monitoring", command=lambda: controller.show_frame(MonitoringPage))
        btn_start_monitoring.pack(pady=50)

        btn_logout = tk.Button(self, text="Logout", command=lambda: controller.show_frame(WelcomePage))
        btn_logout.pack(side=tk.BOTTOM, pady=10)

class MonitoringPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        
        self.controller = controller
        self.display_label = tk.Label(self)
        self.display_label.pack()

        btn_start = tk.Button(self, text="Start Monitoring", command=self.start_monitoring)
        btn_start.pack(pady=10)

        btn_stop = tk.Button(self, text="Stop Monitoring", command=self.stop_monitoring)
        btn_stop.pack(pady=10)

    def start_monitoring(self):
        global cap, monitoring
        if not monitoring:
            cap = cv2.VideoCapture(0)
            monitoring = True
            self.process_frame()

    def stop_monitoring(self):
        global cap, monitoring
        monitoring = False
        if cap:
            cap.release()
        self.display_label.config(image='')
        self.controller.show_frame(Dashboard)

    def process_frame(self):
        global flag, monitoring
        if not monitoring:
            return
        
        ret, frame = cap.read()
        if ret:
            frame = imutils.resize(frame, width=450)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            subjects = detect(gray, 0)
            
            for subject in subjects:
                shape = predict(gray, subject)
                shape = face_utils.shape_to_np(shape)
                leftEye = shape[lStart:lEnd]
                rightEye = shape[rStart:rEnd]
                leftEAR = eye_aspect_ratio(leftEye)
                rightEAR = eye_aspect_ratio(rightEye)
                ear = (leftEAR + rightEAR) / 2.0
                
                leftEyeHull = cv2.convexHull(leftEye)
                rightEyeHull = cv2.convexHull(rightEye)
                cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
                
                if ear < thresh:
                    flag += 1
                    if flag >= frame_check:
                        cv2.putText(frame, "**************** ALERT!****************", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        cv2.putText(frame, "****************ALERT!****************", (10, 325),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        if not mixer.music.get_busy():  # Play sound alert
                            mixer.music.play()
                else:
                    flag = 0
            
            
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)
            self.display_label.imgtk = imgtk
            self.display_label.config(image=imgtk)
        
     
        self.display_label.after(10, self.process_frame)

if __name__ == "__main__":
    app = Application()
    app.mainloop()

