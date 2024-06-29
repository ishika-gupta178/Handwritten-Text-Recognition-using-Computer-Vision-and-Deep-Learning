import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import time
from data_loader import data_loader
import config
from predict import load_easter_model
import cv2
import numpy as np
import itertools
import easter_model


def preprocess(img):            
    # preprocessing is needed 
    img = img/255
    img = img.swapaxes(-2,-1)[...,::-1]
    target = np.ones((config.INPUT_WIDTH, config.INPUT_HEIGHT))
    new_x = config.INPUT_WIDTH/img.shape[0]
    new_y = config.INPUT_HEIGHT/img.shape[1]
    min_xy = min(new_x, new_y)
    new_x = int(img.shape[0]*min_xy)
    new_y = int(img.shape[1]*min_xy)
    img2 = cv2.resize(img, (new_y,new_x))
    target[:new_x,:new_y] = img2
    return 1 - (target)

def decoder(output,letters):
    ret = []
    for j in range(output.shape[0]):
        out_best = list(np.argmax(output[j,:], 1))
        out_best = [k for k, g in itertools.groupby(out_best)]
        outstr = ''
        for c in out_best:
            if c < len(letters):
                outstr += letters[c]
        ret.append(outstr)
    return ret

## rewritten class file for efficient prediction

class infer_img:
    def __init__(self, charlist):
        # self.model = load_easter_model(config.BEST_MODEL_PATH)
        self.model = easter_model.Easter2(predict_mode=True)
        weights_path = r'C:\Users\tiend\OneDrive\Documents\ASU MSBA 2024\CIS515 AI and Data Analytics Strategy\Final Project\Easter2\weights\saved_weights.weights.h5'
        self.model.load_weights(weights_path) 
        self.charlist = charlist
        self.model.summary()
        
    def predict(self, img_path):
        img1 = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        # min_gray, max_gray = img1.min(), img1.max() 
        # constrast_adj = 255 / (max_gray - min_gray) # parameter for constrast adjustment
        # brightness_adj = min_gray # parameter for brightness adjustment
        # adjusted_img = cv2.convertScaleAbs(img1, alpha=constrast_adj, beta=brightness_adj) # increase constrast & brightness

        adjusted_img = img1
        img = preprocess(adjusted_img)
        img = np.expand_dims(img,  0)
        output = self.model.predict(img)
        prediction = decoder(output, self.charlist)
        return prediction

def main_action():
    # Ask user to select the file to upload
    filepath = filedialog.askopenfilename(title="Select the Image File", filetypes=[(".PNG", "*.PNG")])
    result = infer_obj.predict(filepath)

    text_display.insert(tk.END, result)
    text_display.insert(tk.END, '\n')
    text_display.see('end')
    text_display.update_idletasks()
    

if __name__ == "__main__":
    # initialize infer_obj
    with open(r'C:\Users\tiend\OneDrive\Documents\ASU MSBA 2024\CIS515 AI and Data Analytics Strategy\Final Project\Easter2\data\charlist') as f:
        charlist = [word.strip("\n") for word in f ]
    infer_obj = infer_img(charlist)

    # create the app GUI
    root = tk.Tk()
    root.title('Demo')
    ws = root.winfo_screenwidth() # width of the screen
    hs = root.winfo_screenheight() # height of the screen
    w, h = 800, 300 # width and height for the main window
    x = int(ws/2) - int(w/2) # x coordinate to center main window
    y = int(hs/2) - int(h/2) # y coordinate to center main window
    root.geometry(f'{w}x{h}+{x}+{y}') # set geometry of main window
    root.minsize(width=w, height=h) # set minimum size of main window
    predict_button = tk.Button(root, text='Predict Text', command=main_action)
    predict_button.grid(row=0, column=0, padx=5, sticky='NW')
    text_display = tk.Text(root, height=30, width=102)
    text_display.grid(row=0, column=1, padx=5, sticky='W')

    # run the root window
    root.mainloop()
    
    # ## usage
    # ## python predict_line.py --path ~/garbage/images/test1.jpg

    # with open(r'C:\Users\tiend\OneDrive\Documents\ASU MSBA 2024\CIS515 AI and Data Analytics Strategy\Final Project\Easter2\data\charlist') as f:
    #     charlist = [word.strip("\n") for word in f ]
    
    # ## instead of loading the data path, you can write the charlist file in your local storage for the next time.
    # infer_obj = infer_img(charlist)

    # # print(infer_obj.predict(args.path)) ## change the image path with the file path you want
    # # filepath = r'C:\Users\tiend\OneDrive\Documents\ASU MSBA 2024\CIS515 AI and Data Analytics Strategy\Final Project\Easter2\tien.png'
    # filepath = r'C:\Users\tiend\OneDrive\Documents\ASU MSBA 2024\CIS515 AI and Data Analytics Strategy\Final Project\Easter2\test.png'   

    # print(infer_obj.predict(filepath)) ## change the image path with the file path you want