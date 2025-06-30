import os
import cv2
import time
import operator
import numpy as np
import tkinter as tk
from string import ascii_uppercase
from PIL import Image, ImageTk
from keras.models import model_from_json
from spellchecker import SpellChecker


class Application:
    def __init__(self):
        self.vs = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.current_image = None
        self.current_image2 = None

        self.load_models()

        self.ct = {i: 0 for i in ascii_uppercase}
        self.ct['blank'] = 0
        self.blank_flag = 0

        print("Models loaded successfully.")

        self.root = tk.Tk()
        self.root.title("Sign Language To Text Conversion")
        self.root.geometry("900x900")
        self.root.protocol('WM_DELETE_WINDOW', self.destructor)

        self.setup_ui()
        self.video_loop()

    def load_model_from_json(self, model_json_path, model_weights_path):
        with open(model_json_path, 'r') as json_file:
            model_json = json_file.read()
        model = model_from_json(model_json)
        model.load_weights(model_weights_path)
        return model

    def load_models(self):
        model_dir = os.path.join("Models")
        self.loaded_model = self.load_model_from_json(
            os.path.join(model_dir, "model_new.json"),
            os.path.join(model_dir, "model_new.h5")
        )
        self.loaded_model_dru = self.load_model_from_json(
            os.path.join(model_dir, "model-bw_dru.json"),
            os.path.join(model_dir, "model-bw_dru.h5")
        )
        self.loaded_model_tkdi = self.load_model_from_json(
            os.path.join(model_dir, "model-bw_tkdi.json"),
            os.path.join(model_dir, "model-bw_tkdi.h5")
        )
        self.loaded_model_smn = self.load_model_from_json(
            os.path.join(model_dir, "model-bw_smn.json"),
            os.path.join(model_dir, "model-bw_smn.h5")
        )

    def setup_ui(self):
        self.panel = tk.Label(self.root)
        self.panel.place(x=100, y=10, width=580, height=580)

        self.panel2 = tk.Label(self.root)
        self.panel2.place(x=400, y=65, width=275, height=275)

        tk.Label(self.root, text="Sign Language To Text Conversion", font=("Courier", 30, "bold")).place(x=60, y=5)

        self.panel3 = tk.Label(self.root)
        self.panel3.place(x=500, y=540)

        tk.Label(self.root, text="Character :", font=("Courier", 30, "bold")).place(x=10, y=540)
        self.panel4 = tk.Label(self.root)
        self.panel4.place(x=220, y=595)

        tk.Label(self.root, text="Word :", font=("Courier", 30, "bold")).place(x=10, y=595)
        self.panel5 = tk.Label(self.root)
        self.panel5.place(x=350, y=645)

        tk.Label(self.root, text="Sentence :", font=("Courier", 30, "bold")).place(x=10, y=645)
        tk.Label(self.root, text="Suggestions :", fg="red", font=("Courier", 30, "bold")).place(x=250, y=690)

        self.bt1 = tk.Button(self.root, command=self.action1)
        self.bt1.place(x=26, y=745)

        self.bt2 = tk.Button(self.root, command=self.action2)
        self.bt2.place(x=325, y=745)

        self.bt3 = tk.Button(self.root, command=self.action3)
        self.bt3.place(x=625, y=745)

        self.str = ""
        self.word = ""
        self.current_symbol = "Empty"
        self.photo = "Empty"

    #Suggestions 
    
    def get_suggestions(self):
        spell = SpellChecker()
        word = self.word.strip()

        if not word or not word.isalpha():
            return []

        try:
            candidates = spell.candidates(word)
            if candidates is None:
                return []
            suggestions = sorted(candidates, key=lambda w: spell.word_probability(w), reverse=True)
            return suggestions[:3]
        except Exception as e:
            print(f"SpellChecker error: {e}")
            return []

    # def get_suggestions(self):
    #     spell = SpellChecker()
    #     word = self.word.strip()
    #     if not word:
    #         return []
    #     suggestions = list(spell.candidates(word))
    #     suggestions.sort()  # Alphabetical, or use a custom sort
    #     return suggestions



    def video_loop(self):
        ok, frame = self.vs.read()
        if ok:
            cv2image = cv2.flip(frame, 1)
            x1, y1 = int(0.5 * frame.shape[1]), 10
            x2, y2 = frame.shape[1] - 10, int(0.5 * frame.shape[1])
            cv2.rectangle(frame, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (255, 0, 0), 1)

            roi = frame[y1:y2, x1:x2]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 2)
            th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 11, 2)
            _, res = cv2.threshold(th3, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            self.predict(res)
            self.ct[self.current_symbol] += 1


            imgtk = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            self.panel.imgtk = imgtk
            self.panel.config(image=imgtk)

            imgtk2 = ImageTk.PhotoImage(image=Image.fromarray(res))
            self.panel2.imgtk = imgtk2
            self.panel2.config(image=imgtk2)

            self.panel3.config(text=self.current_symbol, font=("Courier", 30))
            self.panel4.config(text=self.word, font=("Courier", 30))
            self.panel5.config(text=self.str, font=("Courier", 30))

            suggestions = self.get_suggestions()
            self.bt1.config(text=suggestions[0] if len(suggestions) > 0 else "", font=("Courier", 20))
            self.bt2.config(text=suggestions[1] if len(suggestions) > 1 else "", font=("Courier", 20))
            self.bt3.config(text=suggestions[2] if len(suggestions) > 2 else "", font=("Courier", 20))

        self.root.after(5, self.video_loop)

    def predict(self, test_image):
        
        test_image = cv2.resize(test_image, (128, 128)).reshape(1, 128, 128, 1).astype('float32') / 255.0
        result = self.loaded_model.predict(test_image)
        result_dru = self.loaded_model_dru.predict(test_image)
        result_tkdi = self.loaded_model_tkdi.predict(test_image)
        result_smn = self.loaded_model_smn.predict(test_image)


        prediction = {'blank': result[0][0]}
        for i, char in enumerate(ascii_uppercase, start=1):
            prediction[char] = result[0][i]

        prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
        self.current_symbol = prediction[0][0]

        if self.current_symbol in ['D', 'R', 'U']:
            pr = dict(zip(['D', 'R', 'U'], result_dru[0]))
            self.current_symbol = max(pr.items(), key=operator.itemgetter(1))[0]

        if self.current_symbol in ['D', 'I', 'K', 'T']:
            pr = dict(zip(['D', 'I', 'K', 'T'], result_tkdi[0]))
            self.current_symbol = max(pr.items(), key=operator.itemgetter(1))[0]

        if self.current_symbol in ['M', 'N', 'S']:
            pr = dict(zip(['M', 'N', 'S'], result_smn[0]))
            self.current_symbol = max(pr.items(), key=operator.itemgetter(1))[0]

        self.ct[self.current_symbol] += 1

        if self.ct[self.current_symbol] > 20:
            if self.current_symbol == 'blank':
                if self.blank_flag == 0:
                    self.blank_flag = 1
                    if self.word:
                        self.str += " " + self.word
                        self.word = ""
            else:
                self.blank_flag = 0
                self.word += self.current_symbol

            self.ct = {i: 0 for i in ascii_uppercase}
            self.ct['blank'] = 0

    def action1(self):
        suggestions = self.get_suggestions()
        if suggestions:
            self.word = ""
            self.str += " " + suggestions[0]

    def action2(self):
        suggestions = self.get_suggestions()
        if len(suggestions) > 1:
            self.word = ""
            self.str += " " + suggestions[1]

    def action3(self):
        suggestions = self.get_suggestions()
        if len(suggestions) > 2:
            self.word = ""
            self.str += " " + suggestions[2]

    def destructor(self):
        print("Closing Application...")
        self.root.destroy()
        self.vs.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    print("Starting Application...")
    Application().root.mainloop()
