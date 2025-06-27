# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 14:01:14 2025

@author: alko18
"""

import os

os.chdir(r'C:\Users\akorn\Desktop\Charié\BA\final_version\PPG_EVA-tool')#set working directory

import tkinter as tk
from tkinter import ttk, filedialog
import sys
import threading

import PPG_EVA_tool as eva

import configparser

stop_event = threading.Event()


def main_loop(testrun = False):
    stop_event.clear()
    if testrun:
        eva.process(stop_event, train=False, testrun=True)    
    else:
        threading.Thread(target = eva.process, args=(stop_event, False, False, False), daemon=True).start()
        


class ConsoleRedirector:
    """Fängt print()-Ausgaben ab und leitet sie an ein Text-Widget weiter"""
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, message):
        """Schreibt den Text ins Textfeld"""
        self.text_widget.insert(tk.END, message)
        self.text_widget.see(tk.END)
        self.text_widget.update_idletasks()# Automatisches Scrollen

    def flush(self):
        """Wird für Kompatibilität benötigt (z.B. bei `sys.stdout.flush()`)"""
        pass

def set_values():
    
    config = configparser.ConfigParser()

    config['Paths'] = {
        'working_folder': working_folder ,
        'export_path': exportpath,
        'classifier_path': r'', #classifier
        'training_values_path': r'' #all_data_tupöes
    }

    config['Settings'] = {
        'fs_A': fs_A,
        'order': order,
        'lowcut': lowcut,
        'highcut': highcut,
        'chunk_length': chunk_length,
        
    }

    with open(r'', 'w') as configfile: #path to configfile
        config.write(configfile)

def main_window(): 
    def directory():

        def search_directory(entry):
            directory_path=filedialog.askdirectory(title='Chose a Folder')
            entry.insert(0, directory_path)
             
        def enable_training():
            """Aktiviert oder deaktiviert die Eingabefelder basierend auf dem Checkbutton-Status."""
            
            if training_var.get():
            
                directory_entry.config(state="disabled")
                directorysearch_button.config(state="disabled")
                export_entry.config(state="disabled")
                exportsearch_button.config(state="disabled")
                
                start_button.config(command=lambda: eva.process(stop_event, train=True))
                
            else:
                "disabled"
               
                directory_entry.config(state="normal")
                directorysearch_button.config(state="normal")
                export_entry.config(state="normal")
                exportsearch_button.config(state="normal")
                       
                start_button.config(command=lambda: main_loop())
                

            
        directory_frame = ttk.Frame(root)
        directory_frame.grid(row=0, column=0, columnspan=2,padx=10, sticky='w')

        label_directory = ttk.Label(directory_frame, text='Directory')
        directory_entry = ttk.Entry(directory_frame, width=100, textvariable=folderpath_val)#folder mit den zu untersuchenden signalen
        directorysearch_button= ttk.Button(directory_frame, text='Search Directory', command=lambda: search_directory(directory_entry))
                
        training_label = ttk.Label(directory_frame, text='Enable Training mode')
        training_var = tk.BooleanVar()
        training_checkbutton=ttk.Checkbutton(directory_frame, variable=training_var, command=enable_training)
        
        export_label = ttk.Label(directory_frame, text='Export')
        export_entry = ttk.Entry(directory_frame, width=100, textvariable=exportpath_val)
        exportsearch_button = ttk.Button(directory_frame, text='Search Directory', command=lambda: search_directory(export_entry))
        
        label_directory.grid(row=0, sticky='w')
        directory_entry.grid(row=1, sticky='w')
        directorysearch_button.grid(row=1, column=1, sticky='w')
        export_label.grid(row=2, column=0, sticky='w')
        export_entry.grid(row=3, column=0, sticky='w')
        exportsearch_button.grid(row=3, column=1, sticky='w')
        training_label.grid(row=4, column=0, sticky='w')
        training_checkbutton.grid(row=4, column=1, sticky='w')



    def metadata():    
        label_MetaData=ttk.Label(root, text='Meatadaten')
        label_MetaData.grid(row=2,padx=10, sticky='w')
        
        metadata_frame = ttk.Frame(root)
        metadata_frame.grid(row=3, column=0, columnspan=2,padx=10, pady=10, sticky='w')
        
        label_fsA = ttk.Label(metadata_frame, text='Samplerate')
        fsA_unit=ttk.Label(metadata_frame, text='Hz')
        fsA_entry = ttk.Entry(metadata_frame, width=10, textvariable=fs_A_val)
        
        label_fsA.grid(row=0, column=0, padx=5, pady=2, sticky='w')
        fsA_entry.grid(row=0, column=1, pady=2, sticky='w')
        fsA_unit.grid(row=0, column=2, sticky='w')
    
    def advanced():

        def filter_():
            filter_label = ttk.Label(advanced_frame, text='Filter')        
            filter_label.grid(row=1, column=0, padx=10, pady=5, sticky='w')
            
            order_label = ttk.Label(filter_frame, text='Order')
            lowcut_label = ttk.Label(filter_frame, text='Lowcut')
            highcut_label = ttk.Label(filter_frame, text='Highcut')
            
            order_entry = ttk.Entry(filter_frame, width=10, textvariable=order_val)
            lowcut_entry = ttk.Entry(filter_frame, width=10, textvariable=lowcut_val)
            highcut_entry = ttk.Entry(filter_frame, width=10, textvariable=highcut_val)
            
            lowcut_unit = ttk.Label(filter_frame, text='Hz')
            highcut_unit = ttk.Label(filter_frame, text='Hz')
            
            order_label.grid(row=1, column=0, padx=10, pady=2, sticky='w')
            lowcut_label.grid(row=2, column=0, padx=10, pady=2,sticky='w')
            highcut_label.grid(row=3, column=0, padx=10, pady=2, sticky='w')
            order_entry.grid(row=1, column=1, pady=2, sticky='w')
            lowcut_entry.grid(row=2, column=1, pady=2, sticky='w')
            highcut_entry.grid(row=3, column=1, pady=2, sticky='w')
            lowcut_unit.grid(row=2, column=2, pady=2, sticky='w')
            highcut_unit.grid(row=3, column=2, pady=2, sticky='w')
            

        def slicing():
            slicing_label = ttk.Label(advanced_frame, text='Slicing')
            slicing_label.grid(row=1, column=1, padx=10, pady=5, sticky='w')

            chunklenght_label=ttk.Label(slicing_frame, text='Chunklength')
            chunklenght_entry=ttk.Entry(slicing_frame, width=10, textvariable=chunk_length_val)
            chunklength_unit=ttk.Label(slicing_frame, text='s')

            chunklenght_label.grid(row=0, column=0, padx=10, pady=5, sticky='w')
            chunklenght_entry.grid(row=0, column=1, pady=5,sticky='w')
            chunklength_unit.grid(row=0, column=2, pady=5,sticky='w')
             
        
        advanced_frame = ttk.Frame(root)
        advanced_frame.grid(row=4, column=0, columnspan=2, sticky='w')
        advanced_label=ttk.Label(advanced_frame, text='Advanced Settings')
        advanced_label.grid(row=0, column=0, padx=10, sticky='w')
        
        filter_frame = ttk.Frame(advanced_frame)
        filter_frame.grid(row=2, column=0, padx=10)
        
        slicing_frame = ttk.Frame(advanced_frame)
        slicing_frame.grid(row=2, column=1, padx=10,)
        
        SNR_frame = ttk.Frame(advanced_frame)
        SNR_frame.grid(row=2, column=2, padx=10,)   
        
        filter_()
        slicing()
        
    def get_values():
       
        global working_folder, exportpath, decision_pfad, training_signal, annotation, fs_A, fs_B, signal_length, order, lowcut, highcut, chunk_length, low_BPM, high_BPM
        working_folder = folderpath_val.get()
        exportpath = exportpath_val.get()
        fs_A = fs_A_val.get()
        order = order_val.get()
        lowcut = lowcut_val.get()
        highcut = highcut_val.get()
        chunk_length = chunk_length_val.get()
        print('Changes applied.')
        
        set_values()
    
    root = tk.Tk()
    root.title('PPG-Eva')
    root.geometry('1060x800')
    root.minsize(width=800, height=800)
    
    folderpath_val = tk.StringVar()
    exportpath_val = tk.StringVar()
    #meta data:
    fs_A_val = tk.IntVar(value=128)
    #filter
    order_val=tk.IntVar(value=2)
    lowcut_val=tk.DoubleVar(value=0.5)
    highcut_val=tk.DoubleVar(value=8.0)
    #slicing
    chunk_length_val=tk.DoubleVar(value=10)
    
    directory()
    
    metadata()
    
    advanced()
    
    apply_button= ttk.Button(root, text='Apply changes', command=get_values)
    apply_button.grid(row=5, column=0, padx=10, pady=10, sticky='w')

    test_button = ttk.Button(root, text='Test', command=lambda: main_loop(True))
    test_button.grid(row=5, column=0, padx=100, pady=5, sticky='w')

    console_label = ttk.Label(root, text='Kernel Output')
    console_label.grid(row=6, column=0,padx=10, sticky='w')
    
    console_output = tk.Text(root, height=10, width=80, wrap='word', state='normal')
    console_output.grid(row=7, column=0, padx=10, pady=5, sticky='w')
    sys.stdout = ConsoleRedirector(console_output)
    
    start_button=ttk.Button(root, text='Start', command=lambda: main_loop())
    start_button.grid(row=8, column=0, padx=10, pady=5, sticky='w')
    abort_button = ttk.Button(root, text='Abort', command=lambda: stop_event.set())
    abort_button.grid(row=8, column=0, padx=100, pady=5, sticky='w')


    root.mainloop()

main_window()





















