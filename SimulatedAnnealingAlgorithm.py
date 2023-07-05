import numpy as np
import pandas as pd
import time                    
from tkinter import *
from tkinter import messagebox
import pandas as pd
from tkinter import filedialog as fd, ttk
import customtkinter

start_time = time.time()                 
np.random.seed(40)

class create_gui():
    def __init__(self):
        super().__init__()
        self.root = customtkinter.CTk()
        self.y = None
        self.x = None
        self.openfile_tree = None
        self.df = None
        self.df1 = None
        self.numeric_columns = None
        self.extract_data_tree = None
        self.labels = None
        self.file = None
        self.current_solution = None
        self.distance = None
        self.textbox2= customtkinter.CTkTextbox(self.root,border_width=1, width=600)
        self.createTreeView()
        self.createTextbox()
        self.createopenFileButton()
        self.runProgramButton()
        self.showResult()
        self.root.mainloop()

    def createopenFileButton(self):
        Button(text="OPEN FILE CSV", command=self.open_csv).grid(row=7, column=1,pady=50)

    def createTreeView(self):
        self.openfile_tree = ttk.Treeview()
        self.openfile_tree.grid(row=7, column=2, padx=50, pady=20)  

    def runProgramButton(self):
        Button(text="RUN PROGRAM", command=self.algorithm, borderwidth=1, width=20).grid(row=8, column=1)  

    def showResult(self):
        Button(text="SHOW RESULT",command=self.show_result,borderwidth=1, width=20).grid(row=8, column=2)

    def createTextbox(self):

        self.textbox2.grid(row=10, column=2,padx=20, pady=20)
    
    def open_csv(self):
        self.file = fd.askopenfilename(title="Open CSV File", filetypes=(("CSV Files", ".csv"), ("All Files", "*.*")))
        if self.x is not None and self.y is not None:
            self.x.destroy()
            self.y.destroy()
        self.df = pd.read_csv(self.file)
        self.df.shape
        
        self.openfile_tree.delete(*self.openfile_tree.get_children())
        self.openfile_tree['column'] = list(self.df.columns)
        self.openfile_tree['show'] = 'headings'

        for col in self.openfile_tree['column']:
            self.openfile_tree.heading(col, text=col)

        df_rows = self.df.to_numpy().tolist()
        for row in df_rows:
            self.openfile_tree.insert("", "end", values=row)
    
    def algorithm(self):
        np.random.seed(40)
        self.df = pd.read_csv(self.file)
        dataset = self.df     
        cities=np.array(dataset)

        def calculate_distance(cities , solution):
            solution_for_distance_calculation = np.append(solution, [solution[0]], axis=0) 
            distance = 0
            next_city_index_founder=0 
            
            for i in solution_for_distance_calculation: 
                next_city_index_founder += 1
                if next_city_index_founder < len(solution_for_distance_calculation):
                    next_city_index=solution_for_distance_calculation[next_city_index_founder] 
                    distance += np.sqrt(((cities[next_city_index,0]-cities[i,0])**2)+((cities[next_city_index,1]-cities[i,1])**2)) 
            return distance        

        def generate_solution(current_solution): 
            idx1 , idx2 = np.random.choice(len(current_solution),2)
            current_solution_copy = current_solution.copy()
            current_solution_copy[idx2], current_solution_copy[idx1] = current_solution_copy[idx1], current_solution_copy[idx2]  
            return current_solution_copy

        def main(dataset, T, cooling_rate, T_lower_bound, tolerance):
            
            current_solution = np.random.permutation(range(len(dataset))) 
            h=0 
            
            while T>T_lower_bound: 
                h+=1
                while True: 
                    potential_solution = generate_solution(current_solution) 
                    potential_distance = calculate_distance(cities , potential_solution) 
                    current_distance = calculate_distance(cities , current_solution)
                
                    if potential_distance < current_distance: 
                        current_solution = potential_solution

                    elif np.random.random() < np.exp(-(potential_distance - current_distance)/T): 
                        current_solution = potential_solution

                    if np.abs(potential_distance-current_distance) < tolerance: 
                        break

                T = T*cooling_rate 

            messagebox.showinfo("Notification","DONE!!!")


            self.current_solution = np.append(current_solution, [current_solution[0]], axis=0)
            self.distance = current_distance
            self.iterations = h            
            self.comp_time = time.time() - start_time     
            
        main(dataset,1000,0.999, 0.01, 1)   
    
    def show_result(self):
        tour = "Best tour founded for salesman:{} \n".format(self.current_solution)
        distance = "Distance of tour founded:{} \n".format(self.distance)
        iteration = "Iterations: {} \n".format(self.iterations)
        time = "Computational Time: {} seconds \n".format(self.comp_time)
        self.textbox2.insert("current",tour)
        self.textbox2.insert("current",distance)
        self.textbox2.insert("current",iteration)
        self.textbox2.insert("current",time)

do = create_gui()

