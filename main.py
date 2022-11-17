import tkinter as tk
from tkinter import filedialog

from Dataprocessor import Dataprocessor
from RBFN import Model
from KMeans import KMeans
from simulator import Simulator

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.patches as patches


def openfile(route_name):
    dataset_url = filedialog.askopenfilename(title="Select file")
    route_name.set(dataset_url.split('/')[-1])

    return dataset_url

def draw_track(f,canvas,route_name):
    global car_x,car_y,phi,final_coordinate,f_plot

    f.clear()
    f_plot = f.add_subplot(111)
    f_plot.clear()

    track_data = Dataprocessor.readfile(openfile(route_name))
    track_data = [data.strip().split(',') for data in track_data]
    track_data = Dataprocessor.text_to_numlist(track_data)
    #初始點，初始角度
    car_x,car_y,phi =  track_data[0][0],track_data[0][1],track_data[0][2]
    #初始線
    final_coordinate = track_data[1:3]
    #賽道牆壁
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

   
def train_model(epochs,learning_rate,window,epoch,loss):
    global rbfModel,feature_len,y_train
    dataprocessor = Dataprocessor()
    x_train,y_train = dataprocessor.splitFile(file_url)
    feature_len = len(x_train[0])
    kmeans = KMeans(x_train,len(x_train[0]))
    m,sigma = kmeans.process()
    print(m,sigma)
    
    rbfModel = Model(epochs,learning_rate,m,sigma,len(x_train[0]))
    rbfModel.train(x_train,y_train,window,epoch,loss)

def print_result(window,canvas,front_distance,right_distance,left_distance):
    simu = Simulator(0,0,90)
    four_dimension,six_dimension = simu.start(window,canvas,f_plot,feature_len,rbfModel,front_distance,right_distance,left_distance)
    if feature_len == 3:
        save_4d_result(four_dimension)
    else:
        save_6d_result(six_dimension)

def save_4d_result(four_dimension):
    file = open("track4D.txt", "w")
    for i in range(len(four_dimension)):
        towrite = ' '.join(str(item) for item in four_dimension[i])
        file.write(f"{towrite}\n")
    file.close

def save_6d_result(six_dimension):
    file = open("track6D.txt", "w")
    for i in range(len(six_dimension)):
        towrite = ' '.join(str(item) for item in six_dimension[i])
        file.write(f"{towrite}\n")
    file.close

def get_file_url(file_name):
    global file_url 
    file_url = filedialog.askopenfilename()
    
    file_name.set(file_url.split('/')[-1])
    
    print(file_url)

def main():
    window = tk.Tk()

    f = Figure(figsize=(5, 4), dpi=100)
    plot_view = tk.Frame(window)
    plot_view.place(x=300,y=20)
    canvas = FigureCanvasTkAgg(f, plot_view)
    canvas.get_tk_widget().pack(side=tk.RIGHT, expand=1)
    
    window.geometry("1000x500+200+300")

    window.title('類神經網路-作業二')
    #選擇檔案
    file_name = tk.StringVar()   # 設定 text 為文字變數
    file_name.set('')            # 設定 text 的內容

    route_name = tk.StringVar()   # 設定 text 為文字變數
    route_name.set('')            # 設定 text 的內容    

    tk.Label(window, text='輸入訓練資料').place(x = 20,y = 20)

    tk.Button(window, text='選擇檔案',command= lambda: get_file_url(file_name)).place(x = 120,y = 16)

    tk.Label(window, text='地圖資料路徑').place(x = 20,y = 50)

    tk.Button(window, text='選擇檔案', command= lambda: draw_track(f,canvas,route_name)).place(x = 120,y = 50)

    tk.Label(window, textvariable=file_name).place(x=180, y=20)

    tk.Label(window, textvariable=route_name).place(x=180, y=50)

    #輸入迭代次數
    tk.Label(window, text='Epoch:').place(x = 20,y = 80)
    interation = tk.Entry(window)
    interation.place(x = 120,y = 80)

    #學習率
    tk.Label(window, text='Learning rate:').place(x = 20,y = 110)
    learning_rate = tk.Entry(window)
    learning_rate.place(x = 120,y = 110)

    #顯示epoch
    epoch = tk.StringVar() 
    epoch.set('')            
    tk.Label(window, text='epoch:').place(x = 20,y = 200)
    tk.Label(window, textvariable=epoch).place(x=120, y=200)
    #顯示loss
    loss = tk.StringVar() 
    loss.set('')            
    tk.Label(window, text='loss:').place(x = 20,y = 230)
    tk.Label(window, textvariable=loss).place(x=120, y=230)


    #開始訓練資料
    tk.Button(window, text='Train',command= lambda: train_model(
            int(interation.get()),
            float(learning_rate.get()),
            window,
            epoch,
            loss
    )).place(x = 120,y = 140)

    front_distance = tk.StringVar() 
    front_distance.set('')            
    tk.Label(window, text='前方感測器距離:').place(x = 20,y = 260)
    tk.Label(window, textvariable=front_distance).place(x=120, y=260)

    right_distance = tk.StringVar() 
    right_distance.set('')
    tk.Label(window, text='右方感測器距離:').place(x = 20,y = 290)
    tk.Label(window, textvariable=right_distance).place(x=120, y=290)

    left_distance = tk.StringVar() 
    left_distance.set('')
    tk.Label(window, text='左方感測器距離:').place(x = 20,y = 320)
    tk.Label(window, textvariable=left_distance).place(x=120, y=320)


    tk.Button(window, text='Start', command= lambda: print_result(window,canvas,front_distance,right_distance,left_distance)).place(x = 120,y = 170)



    window.mainloop()

if __name__=="__main__":
    main()
    