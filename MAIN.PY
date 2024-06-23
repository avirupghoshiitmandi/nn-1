import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
import tkinter as tk
test_data = pd.read_csv("test.csv")
test_data=np.array(test_data).T
# Define the Node class
class Node:
    def __init__(self, canvas, x, y, radius):
        self.canvas = canvas
        self.x = x
        self.y = y
        self.radius = radius
        self.intensity = 0.0  # Default intensity
        self.oval_id = None
        
    def draw(self):
        # Map intensity to yellow color range (light yellow to bright yellow)
        yellow_value = int(self.intensity * 255)
        fill_color = f"#{yellow_value:02x}{yellow_value:02x}00"
        
        x0, y0 = self.x - self.radius, self.y - self.radius
        x1, y1 = self.x + self.radius, self.y + self.radius
        
        if self.oval_id:
            self.canvas.delete(self.oval_id)
            
        self.oval_id = self.canvas.create_oval(x0, y0, x1, y1, outline='black', fill=fill_color)
        
    def set_intensity(self, intensity):
        self.intensity = max(0.0, min(1.0, intensity))  # Ensure intensity is between 0 and 1
        self.draw()
        
    def change_color(self, color):
        x0, y0 = self.x - self.radius, self.y - self.radius
        x1, y1 = self.x + self.radius, self.y + self.radius
        
        if self.oval_id:
            self.canvas.delete(self.oval_id)
            
        self.oval_id = self.canvas.create_oval(x0, y0, x1, y1, outline='black', fill=color)

# Function to load neural network parameters from .npy files
def load_parameters():
    W1 = np.load('W1.npy')
    B1 = np.load('B1.npy')
    W2 = np.load('W2.npy')
    B2 = np.load('B2.npy')
    W3 = np.load('W3.npy')
    B3 = np.load('B3.npy')
    return W1, B1, W2, B2, W3, B3

# Function for forward propagation through the neural network
def forward_prop(W1, B1, W2, B2, W3, B3, X):
    X = X.reshape((-1, 1))
    z1 = W1.dot(X) + B1
    A1 = ReLU(z1)
    print
    z2 = W2.dot(A1) + B2
    A2 = ReLU(z2)
    
    z3 = W3.dot(A2) + B3
    A3 = softmax(z3)
    
    A1_MAX = np.argmax(A1, 0)
    A2_MAX = np.argmax(A2, 0)
    A3_MAX = np.argmax(A3, 0)
    
    return z1, A1, z2, A2, z3, A3, A1_MAX, A2_MAX, A3_MAX

# Activation functions
def ReLU(Z):
    return np.maximum(0, Z)

def softmax(Z):
    expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return expZ / np.sum(expZ, axis=0, keepdims=True)

# GUI application class
class NodeDrawingApp:
    def __init__(self, master): 
        self.master = master
        self.master.title("DIGIT CHECKER")
        
        self.canvas = tk.Canvas(self.master, width=1000, height=1000, bg='white')
        self.canvas.pack(pady=20)
        
        # Lists to hold nodes
        self.nodesin=[]
        self.nodes1 = []
        self.nodes2 = []
        self.nodesout = []
        
        # Create nodes on canvas
        
        
        # Load neural network parameters
        W1, B1, W2, B2, W3, B3 = load_parameters()
        
        # Example: Update nodes based on predictions
        for i in range(100):  # Update for 10 test cases (adjust as needed)
            
            A1_MAX, A2_MAX, A3_MAX ,current_image= prediction(i, W1, B1, W2, B2, W3, B3)
            self.create_nodes()
            self.update_nodes(A1_MAX, A2_MAX, A3_MAX)
            print("Prediction: ", A3_MAX)
            current_image = current_image.reshape((28, 28)) * 255
            plt.gray()
            plt.imshow(current_image, interpolation='nearest')
            plt.show()
         
    def create_nodes(self):
        self.canvas.delete('all')
        radius = 10
        x = 300
        y = 50
        
        # Create input layer nodes
        for i in range(5):
            node = Node(self.canvas, x - 100, y + i * 30, radius)
            self.nodesin.append(node)
            node.draw()
        
        for i in range(5):
            node = Node(self.canvas, x - 100, y + 450 + i * 30, radius)
            self.nodesin.append(node)
            node.draw()
        
        # Create hidden layer nodes
        for i in range(20):
            node = Node(self.canvas, x, y + i * 30, radius)
            self.nodes1.append(node)
            node.draw()
        
        for i in range(20):
            node = Node(self.canvas, x + 100, y + i * 30, radius)
            self.nodes2.append(node)
            node.draw()
        self.draw_dots()
        # Create output layer nodes
        for i in range(10):
            node = Node(self.canvas, x + 200, 200 + i * 30, radius)
            self.nodesout.append(node)
            node.draw()
            self.canvas.create_text(x + 200, 200 + i * 30, text=f"{i}", font=('Arial', 12), fill='yellow')
    def draw_dots(self):
        # Coordinates and colors for dots
        dots = []
        for i in range(1,15):
            dots.append((200,180 + i*20))
        
        for x, y in dots:
            self.canvas.create_oval(x-3, y-3, x+3, y+3, fill='black', outline='')    
    def update_nodes(self, A1_MAX, A2_MAX, A3_MAX):
        # Reset colors for all nodes
   
        # Set color for nodes based on predictions'
        
        self.nodes1[int(A1_MAX)].change_color('yellow')
        self.nodes2[int(A2_MAX)].change_color('yellow')
        self.nodesout[int(A3_MAX)].change_color('blue')
        self.canvas.create_text(self.nodesout[int(A3_MAX)].x,self.nodesout[int(A3_MAX)].y,  text=f"{int(A3_MAX)}", font=('Arial', 12), fill='yellow')

        for i in self.nodesin:
         self.draw_line(i, self.nodes1[int(A1_MAX)])
        self.draw_line(self.nodes1[int(A1_MAX)], self.nodes2[int(A2_MAX)])
        self.draw_line(self.nodes2[int(A2_MAX)], self.nodesout[int(A3_MAX)])
    def draw_line(self, node1, node2):
        x1, y1 = node1.x, node1.y
        x2, y2 = node2.x, node2.y
        self.canvas.create_line(x1, y1, x2, y2, fill='blue')    

# Function to make predictions (example usage)
def prediction(index, W1, B1, W2, B2, W3, B3):
    current_image = test_data[:, index]
    
    _, _, _, _, _, _, A1_MAX, A2_MAX, A3_MAX = forward_prop(W1, B1, W2, B2, W3, B3, current_image)
    print(A3_MAX)
    return A1_MAX, A2_MAX, A3_MAX,current_image

def main():
    root = tk.Tk()
    app = NodeDrawingApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
