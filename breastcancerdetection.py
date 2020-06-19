from tkinter import *
import urllib3, requests, json
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import random

window = Tk()
window.geometry("900x600")
window.title("Breast Cancer Detection using AI and ML")

#=================== text variable =================================
radiusVar= StringVar()
textureVar = StringVar()
perimeterVar= StringVar()
areaVar = StringVar()
smoothnessVar = StringVar()
compactnessVar = StringVar()
concavityVar = StringVar()
concaveVar = StringVar()
symmetryVar = StringVar()
factalVar = StringVar()
displaydataText = Text(window,height=15, width=100)

def getData():
    try:
        radius = float(radiusVar.get())
        texture = float(textureVar.get())
        perimeter = float(perimeterVar.get())
        area = float(areaVar.get())
        smoothness = float(smoothnessVar.get())
        compactness = float(compactnessVar.get())
        concavity = float(concavityVar.get())
        concave = float(concaveVar.get())
        symmetry = float(symmetryVar.get())
        factal = float(factalVar.get())
        
        data=[radius, texture, perimeter, area, smoothness, compactness, concavity, concave, symmetry, factal]
        displaydataText.delete("1.0","end")
        displaydataText.insert(INSERT,"Detecting breast Cancer......")
        df_data=pd.read_csv("requireddata.csv")
        labelencoder_y = LabelEncoder()
        
        start=random.randint(0,300)
        b= random.randint(100,200)
        stop = start+b
        df_data.iloc[:,0]=labelencoder_y.fit_transform(df_data.iloc[:,0].values)
        X_values = df_data.iloc[: ,1:].values
        y_values = df_data.iloc[:,0].values
        x_parameter=X_values[start:stop]
        x_parameter=x_parameter.tolist()
        x_parameter.insert(0, data)
        x_parameter= np.asarray(x_parameter)
        #x_parameters=[]
        #x_parameters.append(x_parameter)
        sc = StandardScaler()
        
        x_train= sc.fit_transform(X_values)
        
        
        test = sc.fit_transform(x_parameter)
        
        forest=RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
        forest.fit(x_train, y_values)
        if(forest.predict(test)[0]==0):
            
            displaydataText.delete("1.0","end")
            displaydataText.insert(INSERT,"Detected\nDon'tWorry. It is Benign Cancer which is not the symbol of Breast Cancer")
        else:
            
            displaydataText.delete("1.0","end")
            displaydataText.insert(INSERT,"Detected\n You have been detected Breast Cancer.\n Following should be taken as precautions:\n1. If you like to drink alcohol you should limit or you should left alcohol.\n2. Please Don't Smoke it make case compicated.\n3. Please Limit dose of hormone therapy if any\n4. Control your weight and please focus in physical activities.\n5. Breast feed you child.\n6. Have plenty of vegetables.")
    except Exception as e:
        displaydataText.delete("1.0","end")
        displaydataText.insert(INSERT,"Please fill Valid data on every fill in decimal number ")
        print(e)
    
    
    
def mainWindow():
    radiusLabel = Label(window, text="Radius Mean")
    radiusLabel.grid(column=0, row=0,padx=(40,0), pady=(30,0))

    radiusEntry= Entry(window, textvariable=radiusVar,width =20)
    radiusEntry.grid(column=0, row=1,padx=(40,0), pady=(5,0))

    textureLabel = Label(window, text="Texture Mean")
    textureLabel.grid(column=1, row=0,padx=(40,0), pady=(30,0))

    textureEntry= Entry(window, textvariable=textureVar,width =20)
    textureEntry.grid(column=1, row=1,padx=(40,0), pady=(5,0))

    perimeterLabel = Label(window, text="Perimeter Mean")
    perimeterLabel.grid(column=2, row=0,padx=(40,0), pady=(30,0))

    perimeterEntry= Entry(window, textvariable=perimeterVar,width =20)
    perimeterEntry.grid(column=2, row=1,padx=(40,0), pady=(5,0))

    areaLabel = Label(window, text="Area Mean")
    areaLabel.grid(column=3, row=0,padx=(40,0), pady=(30,0))

    areaEntry= Entry(window, textvariable=areaVar,width =20)
    areaEntry.grid(column=3, row=1,padx=(40,0), pady=(5,0))

    smoothnessLabel = Label(window, text="Smoothness Mean")
    smoothnessLabel.grid(column=0, row=2,padx=(40,0), pady=(30,0))

    smoothnessEntry= Entry(window, textvariable=smoothnessVar,width =20)
    smoothnessEntry.grid(column=0, row=3,padx=(40,0), pady=(5,0))
    
    compactnessLabel = Label(window, text="Compactness Mean")
    compactnessLabel.grid(column=1, row=2,padx=(40,0), pady=(30,0))

    compactnessEntry= Entry(window, textvariable=compactnessVar,width =20)
    compactnessEntry.grid(column=1, row=3,padx=(40,0), pady=(5,0))

    concavityLabel = Label(window, text="Concavity Mean")
    concavityLabel.grid(column=2, row=2,padx=(40,0), pady=(30,0))

    concavityEntry= Entry(window, textvariable=concavityVar,width =20)
    concavityEntry.grid(column=2, row=3,padx=(40,0), pady=(5,0))

    concaveLabel = Label(window, text="Concave points Mean")
    concaveLabel.grid(column=3, row=2,padx=(40,0), pady=(30,0))
    
    concaveEntry= Entry(window, textvariable=concaveVar,width =20)
    concaveEntry.grid(column=3, row=3,padx=(40,0), pady=(5,0))

    symmetryMean = Label(window, text="Symmetry Mean")
    symmetryMean.grid(column=0, row=4,padx=(40,0), pady=(30,0))
    
    symmetryEntry= Entry(window, textvariable=symmetryVar,width =20)
    symmetryEntry.grid(column=0, row=5,padx=(40,0), pady=(5,0))
    
    FactalLabel = Label(window, text="Factal dimensions Mean")
    FactalLabel.grid(column=1, row=4,padx=(40,0), pady=(30,0))
    factalEntry= Entry(window, textvariable=factalVar,width =20)
    factalEntry.grid(column=1, row=5,padx=(40,0), pady=(5,0))

    PredictButton = Button(window, text="PREDICT", width=20, height=2, command= lambda: getData())
    PredictButton.grid(column=2, row=4, columnspan=2, rowspan=2,pady=(30,0))

    outputText = Label(window, text="OUTPUT")
    outputText.grid(column=0, row=6, padx=(40,0), pady=(60,0))
    
    displaydataText.grid(column=0, row=7, padx=(40,0), pady=(20,0), columnspan=4)
    
mainWindow()

window.mainloop()
