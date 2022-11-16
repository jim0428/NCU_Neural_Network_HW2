import tkinter as tk
from tkinter import filedialog

from Dataprocessor import Dataprocessor
from RBFN import model
from KMeans import KMeans
from simulator import simulator

from simple_playground import *

import time 
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.patches as patches

# import sympy and Point, Line
from sympy import Point, Line

def openfile():
    dataset_url = filedialog.askopenfilename(title="Select file")
    return dataset_url

def draw_track(f,canvas):
    global car_x,car_y,phi,final_coordinate,f_plot
    f.clear()
    f_plot = f.add_subplot(111)
    f_plot.clear()

    track_data = Dataprocessor.readfile(openfile())
    track_data = [data.strip().split(',') for data in track_data]
    track_data = Dataprocessor.text_to_numlist(track_data)

    car_x,car_y,phi =  track_data[0][0],track_data[0][1],track_data[0][2]
    final_coordinate = track_data[1:3]
    data_coordinate = track_data[3:]

    for idx in range(1,len(data_coordinate)):
        x0, y0 = data_coordinate[idx-1][0], data_coordinate[idx - 1][1]
        x1, y1 = data_coordinate[idx][0] , data_coordinate[idx][1]
        #畫線
        f_plot.plot([x0, x1], [y0, y1],'dodgerblue')
    
    #畫初始線
    f_plot.plot([-6,6], [0, 0],'dodgerblue')
        
    #畫矩形
    rect = patches.Rectangle(final_coordinate[0], final_coordinate[1][0] - final_coordinate[0][0], final_coordinate[1][1] - final_coordinate[0][1], fill=True, edgecolor = 'dodgerblue',linewidth = 0)
    f_plot.add_patch(rect)
    canvas.draw()

   


def predict_data(epochs,learning_rate):
    global rbfModel,feature_len,y_train
    dataprocessor = Dataprocessor()
    x_train,y_train = dataprocessor.splitFile(file_url)
    feature_len = len(x_train[0])
    kmeans = KMeans(x_train,len(x_train[0]))
    m,sigma = kmeans.process()
    print(m,sigma)
    
    rbfModel = model(epochs,learning_rate,m,sigma,len(x_train[0]))
    rbfModel.train(x_train,y_train)
    #print(rbfModel.predict(np.array([19.0,  5.48528137,  5.48528137])))


def print_result(f,canvas):
    simu = simulator(0,0,90)
    car_pos = simu.check_collision(y_train,rbfModel)
    for pos in car_pos:
        f_plot.scatter(pos[0],pos[1])
        circle = patches.Circle((pos[0],pos[1]), radius=3, fill=False, color="g")
        f_plot.add_patch(circle)
    canvas.draw()



def save_4d_result(state_list, action_list):
    file = open("4d_result.txt", "w")
    for i in range(len(action_list)):
        str = ""
        file.write(f"{state_list[i]} {action_list[i][0]-40}\n")
    file.close

def save_6d_result(position_list, state_list, action_list):
    file = open("6d_result.txt", "w")
    for i in range(len(action_list)):
        file.write(f"{position_list[i]} {state_list[i]} {action_list[i][0]-40}\n")
    file.close()

def get_file_url(file_name):
    global file_url 
    file_url = filedialog.askopenfilename()
    
    file_name.set(file_url.split('/')[-1])
    
    print(file_url)

def main():
    window = tk.Tk()
    # canvas = tk.Canvas(window, bg='white', height=200, width=200)
    # canvas.pack()
    f = Figure(figsize=(5, 4), dpi=100)

    test = tk.Frame(window)
    test.place(x=300,y=20)
    canvas = FigureCanvasTkAgg(f, test)
    canvas.get_tk_widget().pack(side=tk.RIGHT, expand=1)

    window.geometry("1000x500+200+300")

    window.title('類神經網路-作業二')
    #選擇檔案
    file_name = tk.StringVar()   # 設定 text 為文字變數
    file_name.set('')            # 設定 text 的內容

    tk.Label(window, text='輸入訓練資料').place(x = 20,y = 20)

    tk.Button(window, text='選擇檔案',command= lambda: get_file_url(file_name)).place(x = 120,y = 16)

    tk.Label(window, text='地圖資料路徑').place(x = 20,y = 50)

    tk.Button(window, text='選擇檔案', command= lambda: draw_track(f,canvas)).place(x = 120,y = 50)

    tk.Label(window, textvariable=file_name).place(x=180, y=20)

    #輸入迭代次數
    tk.Label(window, text='Epoch:').place(x = 20,y = 80)
    interation = tk.Entry(window)
    interation.place(x = 120,y = 80)

    #學習率
    tk.Label(window, text='Learning rate:').place(x = 20,y = 110)
    learning_rate = tk.Entry(window)
    learning_rate.place(x = 120,y = 110)

    #開始預測資料
    tk.Button(window, text='Train',command= lambda: predict_data(
            int(interation.get()),
            float(learning_rate.get())
    )).place(x = 80,y = 140)


    tk.Button(window, text='Start', command= lambda: print_result(f,canvas)).place(x = 80,y = 170)


    window.mainloop()

if __name__=="__main__":
    main()
    