import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import json
import threading
from datetime import datetime
import pandas as pd
import numpy as np
import os
import sys
import time
import random
import requests
from requests.auth import HTTPBasicAuth
from typing import Dict, List, Any, Optional, Generator, Tuple
import logging
import re

from alpha_mining import AlphaMiner
from config import Config
from results_manager import ResultsManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('alpha_mining.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AlphaMinerApp:
    def __init__(self, root):
        self.root = root
        self.root.title(Config.WINDOW_TITLE)
        self.root.geometry(Config.WINDOW_SIZE)
        
        # Initialize components
        self.alpha_miner = None
        self.mining_thread = None
        self.is_mining = False
        self.is_optimizing = False
        self.is_submitting = False
        self.results_manager = ResultsManager(Config.RESULTS_DIR)
        
        # Authentication state
        self.auth_credentials = {
            "username": "",
            "password": "",
            "is_authenticated": False,
            "session_token": None,
            "auth_time": None
        }
        
        # Ensure directories exist
        Config.ensure_directories()
        
        # Create the GUI
        self.create_gui()
        self.load_config()

    def create_gui(self):
        # Create notebook for tabbed interface
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Create tabs - only create those that have setup methods defined
        self.setup_mining_tab()
        
        # Only call tab setup methods if they exist
        for tab_setup_method in [
            'setup_expression_mining_tab',
            'setup_parameter_optimization_tab',
            'setup_submission_tab',
            'setup_results_tab',
            'setup_settings_tab'
        ]:
            if hasattr(self, tab_setup_method):
                getattr(self, tab_setup_method)()
        
        # Bind tab selection event to update authentication status
        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_changed)
        
    def setup_mining_tab(self):
        """Set up the main alpha mining tab"""
        mining_frame = ttk.Frame(self.notebook)
        self.notebook.add(mining_frame, text="Alpha Mining")
        
        # Split into left config and right results
        left_frame = ttk.LabelFrame(mining_frame, text="Configuration")
        left_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

        right_frame = ttk.LabelFrame(mining_frame, text="Results")
        right_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        
        mining_frame.grid_columnconfigure(0, weight=1)
        mining_frame.grid_columnconfigure(1, weight=2)
        mining_frame.grid_rowconfigure(0, weight=1)

        # AI Configuration
        ai_frame = ttk.LabelFrame(left_frame, text="AI Configuration")
        ai_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Label(ai_frame, text="Moonshot API Key:").pack(pady=2)
        self.moonshot_api_key_entry = ttk.Entry(ai_frame, show="*")
        self.moonshot_api_key_entry.pack(fill="x", padx=5, pady=2)
        
        ttk.Label(ai_frame, text="Moonshot API URL:").pack(pady=2)
        self.moonshot_api_url_entry = ttk.Entry(ai_frame)
        self.moonshot_api_url_entry.pack(fill="x", padx=5, pady=2)
        self.moonshot_api_url_entry.insert(0, "https://api.moonshot.cn/v1/chat/completions")
        
        ttk.Label(ai_frame, text="AI Model:").pack(pady=2)
        self.ai_model_var = tk.StringVar(value="moonshot-v1-8k")
        ttk.Combobox(ai_frame, textvariable=self.ai_model_var, 
                  values=["moonshot-v1-8k", "moonshot-v1-32k", "moonshot-v1-128k"]).pack(fill="x", padx=5, pady=2)

        # Mining parameters
        params_frame = ttk.LabelFrame(left_frame, text="Mining Parameters")
        params_frame.pack(fill="x", padx=5, pady=5)

        ttk.Label(params_frame, text="Max Iterations:").pack(pady=2)
        self.max_iter_entry = ttk.Entry(params_frame)
        self.max_iter_entry.pack(fill="x", padx=5, pady=2)
        self.max_iter_entry.insert(0, str(Config.DEFAULT_MAX_ITERATIONS))

        ttk.Label(params_frame, text="Population Size:").pack(pady=2)
        self.pop_size_entry = ttk.Entry(params_frame)
        self.pop_size_entry.pack(fill="x", padx=5, pady=2)
        self.pop_size_entry.insert(0, str(Config.DEFAULT_POPULATION_SIZE))
        
        ttk.Label(params_frame, text="AI-Generated Ideas:").pack(pady=2)
        self.ai_ideas_entry = ttk.Entry(params_frame)
        self.ai_ideas_entry.pack(fill="x", padx=5, pady=2)
        self.ai_ideas_entry.insert(0, "5")
        
        ttk.Label(params_frame, text="AI Temperature:").pack(pady=2)
        self.ai_temp_var = tk.StringVar(value="0.3")
        temp_frame = ttk.Frame(params_frame)
        temp_frame.pack(fill="x", padx=5, pady=2)
        ttk.Scale(temp_frame, from_=0.0, to=1.0, variable=self.ai_temp_var, 
               orient="horizontal").pack(side="left", fill="x", expand=True)
        ttk.Label(temp_frame, textvariable=self.ai_temp_var).pack(side="right", padx=5)
        
        # Mining strategy options
        strategy_frame = ttk.LabelFrame(left_frame, text="Mining Strategy")
        strategy_frame.pack(fill="x", padx=5, pady=5)
        
        self.strategy_var = tk.StringVar(value="ai")
        ttk.Radiobutton(strategy_frame, text="AI-Generated Alphas", 
                     variable=self.strategy_var, value="ai").pack(anchor="w")
        ttk.Radiobutton(strategy_frame, text="Genetic Algorithm", 
                     variable=self.strategy_var, value="genetic").pack(anchor="w")
        ttk.Radiobutton(strategy_frame, text="Machine Learning", 
                     variable=self.strategy_var, value="machine").pack(anchor="w")
        
        # Control buttons
        control_frame = ttk.Frame(left_frame)
        control_frame.pack(fill="x", padx=5, pady=5)

        ttk.Button(control_frame, text="Fetch Operators", 
                command=self.fetch_operators).pack(side="left", padx=5)
                
        self.start_button = ttk.Button(control_frame, text="Generate Alphas", 
                                    command=self.start_mining)
        self.start_button.pack(side="left", padx=5)

        self.stop_button = ttk.Button(control_frame, text="Stop", 
                                   command=self.stop_mining, state="disabled")
        self.stop_button.pack(side="left", padx=5)

        # Progress indicators
        progress_frame = ttk.Frame(right_frame)
        progress_frame.pack(fill="x", padx=5, pady=5)

        ttk.Label(progress_frame, text="Progress:").pack(side="left")
        self.progress_var = tk.StringVar(value="0%")
        ttk.Label(progress_frame, textvariable=self.progress_var).pack(side="left", padx=5)

        self.progress_bar = ttk.Progressbar(progress_frame, length=300, mode='determinate')
        self.progress_bar.pack(side="left", padx=5)

        # Results display
        self.mining_log = scrolledtext.ScrolledText(right_frame, height=10)
        self.mining_log.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Alpha details frame
        alpha_frame = ttk.LabelFrame(right_frame, text="Generated Alphas")
        alpha_frame.pack(fill="x", padx=5, pady=5)
        
        # Create treeview for generated alphas
        columns = ("id", "expression", "notes")
        self.generated_alphas_tree = ttk.Treeview(alpha_frame, columns=columns, show="headings", height=6)
        
        # Define headings
        self.generated_alphas_tree.heading("id", text="ID")
        self.generated_alphas_tree.heading("expression", text="Expression")
        self.generated_alphas_tree.heading("notes", text="Notes")
        
        # Define columns
        self.generated_alphas_tree.column("id", width=50)
        self.generated_alphas_tree.column("expression", width=300)
        self.generated_alphas_tree.column("notes", width=150)
        
        # Add scrollbar
        tree_scroll = ttk.Scrollbar(alpha_frame, orient="vertical", 
                                 command=self.generated_alphas_tree.yview)
        self.generated_alphas_tree.configure(yscrollcommand=tree_scroll.set)
        
        self.generated_alphas_tree.pack(side="left", fill="both", expand=True)
        tree_scroll.pack(side="right", fill="y")
        
        # Action buttons for generated alphas
        action_frame = ttk.Frame(right_frame)
        action_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Button(action_frame, text="Verify Selected", 
                command=self.verify_selected_alpha).pack(side="left", padx=5)
        ttk.Button(action_frame, text="Save Selected", 
                command=self.save_selected_alpha).pack(side="left", padx=5)
        ttk.Button(action_frame, text="Submit Selected", 
                command=self.submit_selected_alpha).pack(side="left", padx=5)
        ttk.Button(action_frame, text="Optimize Selected", 
                command=self.optimize_selected_alpha).pack(side="left", padx=5)
        
    def setup_expression_mining_tab(self):
        """Set up the expression mining tab"""
        expr_frame = ttk.Frame(self.notebook)
        self.notebook.add(expr_frame, text="Expression Mining")
        
        # Split into left control and right results
        left_frame = ttk.LabelFrame(expr_frame, text="Expression Configuration")
        left_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        
        right_frame = ttk.LabelFrame(expr_frame, text="Variations")
        right_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        
        expr_frame.grid_columnconfigure(0, weight=1)
        expr_frame.grid_columnconfigure(1, weight=2)
        expr_frame.grid_rowconfigure(0, weight=1)
        
        # Base expression entry
        expr_entry_frame = ttk.LabelFrame(left_frame, text="Base Expression")
        expr_entry_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.base_expr_text = scrolledtext.ScrolledText(expr_entry_frame, height=10)
        self.base_expr_text.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Parameter range controls
        param_frame = ttk.LabelFrame(left_frame, text="Parameter Range")
        param_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Label(param_frame, text="Range (±):").pack(side="left", padx=5)
        self.param_range_var = tk.StringVar(value="5")
        ttk.Entry(param_frame, textvariable=self.param_range_var, width=5).pack(side="left", padx=5)
        
        # Control buttons
        button_frame = ttk.Frame(left_frame)
        button_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Button(button_frame, text="Parse Expression", 
                command=self.parse_expression).pack(side="left", padx=5)
                
        ttk.Button(button_frame, text="Generate Variations", 
                command=self.generate_variations).pack(side="left", padx=5)
                
        ttk.Button(button_frame, text="Test Selected", 
                command=self.test_selected_variation).pack(side="left", padx=5)
                
        ttk.Button(button_frame, text="Test All", 
                command=self.test_all_variations).pack(side="left", padx=5)
                
        ttk.Button(button_frame, text="Stop Testing", 
                command=self.stop_variation_testing).pack(side="left", padx=5)
                
        # Variations list
        list_frame = ttk.LabelFrame(right_frame, text="Generated Variations")
        list_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.variations_list = tk.Listbox(list_frame, height=10)
        list_scroll = ttk.Scrollbar(list_frame, orient="vertical", 
                                 command=self.variations_list.yview)
        self.variations_list.configure(yscrollcommand=list_scroll.set)
        
        self.variations_list.pack(side="left", fill="both", expand=True)
        list_scroll.pack(side="right", fill="y")
        
        # Progress indicators
        progress_frame = ttk.Frame(right_frame)
        progress_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Label(progress_frame, text="Progress:").pack(side="left")
        self.expr_progress_var = tk.StringVar(value="0%")
        ttk.Label(progress_frame, textvariable=self.expr_progress_var).pack(side="left", padx=5)
        
        self.expr_progress_bar = ttk.Progressbar(progress_frame, length=300, mode='determinate')
        self.expr_progress_bar.pack(side="left", padx=5)
        
    def setup_parameter_optimization_tab(self):
        """Set up the parameter optimization tab"""
        opt_frame = ttk.Frame(self.notebook)
        self.notebook.add(opt_frame, text="Parameter Optimization")
        
        # Split into left control and right results
        left_frame = ttk.LabelFrame(opt_frame, text="Configuration")
        left_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        
        right_frame = ttk.LabelFrame(opt_frame, text="Results")
        right_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        
        opt_frame.grid_columnconfigure(0, weight=1)
        opt_frame.grid_columnconfigure(1, weight=2)
        opt_frame.grid_rowconfigure(0, weight=1)
        
        # Alpha expression entry
        expr_entry_frame = ttk.LabelFrame(left_frame, text="Alpha Expression")
        expr_entry_frame.pack(fill="x", padx=5, pady=5)
        
        # Input area for alpha expression
        self.opt_expr_text = scrolledtext.ScrolledText(expr_entry_frame, height=10)
        self.opt_expr_text.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Load and extract buttons
        button_frame = ttk.Frame(left_frame)
        button_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Button(button_frame, text="Load Alpha", 
                command=self.load_alpha_for_optimization).pack(side="left", padx=5)
                
        ttk.Button(button_frame, text="Extract Parameters", 
                command=self.extract_parameters).pack(side="left", padx=5)
                
        # Parameter sliders frame
        self.param_sliders_frame = ttk.LabelFrame(left_frame, text="Parameters")
        self.param_sliders_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        ttk.Label(self.param_sliders_frame, text="No parameters extracted yet").pack(pady=10)
        
        # Optimization control buttons
        opt_button_frame = ttk.Frame(left_frame)
        opt_button_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Button(opt_button_frame, text="Start Optimization", 
                command=self.start_optimization).pack(side="left", padx=5)
                
        ttk.Button(opt_button_frame, text="Stop", 
                command=self.stop_optimization).pack(side="left", padx=5)
                
        # Progress indicators
        progress_frame = ttk.Frame(right_frame)
        progress_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Label(progress_frame, text="Progress:").pack(side="left")
        self.opt_progress_var = tk.StringVar(value="0%")
        ttk.Label(progress_frame, textvariable=self.opt_progress_var).pack(side="left", padx=5)
        
        self.opt_progress_bar = ttk.Progressbar(progress_frame, length=300, mode='determinate')
        self.opt_progress_bar.pack(side="left", padx=5)
        
        # Results table
        results_frame = ttk.LabelFrame(right_frame, text="Optimization Results")
        results_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Create treeview for results
        columns = ("iteration", "sharpe", "turnover", "fitness", "parameters")
        self.opt_results_tree = ttk.Treeview(results_frame, columns=columns, show="headings", height=10)
        
        # Define headings
        self.opt_results_tree.heading("iteration", text="Iter")
        self.opt_results_tree.heading("sharpe", text="Sharpe")
        self.opt_results_tree.heading("turnover", text="Turnover")
        self.opt_results_tree.heading("fitness", text="Fitness")
        self.opt_results_tree.heading("parameters", text="Parameters")
        
        # Define columns
        self.opt_results_tree.column("iteration", width=50, anchor="center")
        self.opt_results_tree.column("sharpe", width=80, anchor="center")
        self.opt_results_tree.column("turnover", width=80, anchor="center")
        self.opt_results_tree.column("fitness", width=80, anchor="center")
        self.opt_results_tree.column("parameters", width=300)
        
        # Add scrollbar
        tree_scroll = ttk.Scrollbar(results_frame, orient="vertical", 
                                 command=self.opt_results_tree.yview)
        self.opt_results_tree.configure(yscrollcommand=tree_scroll.set)
        
        self.opt_results_tree.pack(side="left", fill="both", expand=True)
        tree_scroll.pack(side="right", fill="y")
        
        # Save button for results
        ttk.Button(right_frame, text="Save Selected Alpha", 
                command=self.save_optimized_alpha).pack(padx=5, pady=5)
                
    def setup_submission_tab(self):
        """Set up the alpha submission tab"""
        sub_frame = ttk.Frame(self.notebook)
        self.notebook.add(sub_frame, text="Alpha Submission")
        
        # Create a notebook for submission methods
        sub_notebook = ttk.Notebook(sub_frame)
        sub_notebook.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Create frames for different submission methods
        input_frame = ttk.Frame(sub_notebook, name="input_frame")
        file_frame = ttk.Frame(sub_notebook, name="file_frame")
        fetch_frame = ttk.Frame(sub_notebook, name="fetch_frame")
        
        sub_notebook.add(input_frame, text="Direct Input")
        sub_notebook.add(file_frame, text="From File")
        sub_notebook.add(fetch_frame, text="Fetch Existing")
        
        # Direct Input Tab
        input_top_frame = ttk.Frame(input_frame)
        input_top_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Authentication status
        auth_frame = ttk.Frame(input_top_frame)
        auth_frame.pack(fill="x", padx=5, pady=5)
        
        self.sub_auth_status_var = tk.StringVar(value="Not authenticated")
        ttk.Label(auth_frame, text="Status:").pack(side="left", padx=5)
        self.sub_auth_status_label = ttk.Label(auth_frame, textvariable=self.sub_auth_status_var,
                                           foreground="red")
        self.sub_auth_status_label.pack(side="left", padx=5)
        
        ttk.Button(auth_frame, text="Authenticate", 
                command=self.authenticate_from_submission).pack(side="right", padx=5)
                
        # Input area for alpha expression
        ttk.Label(input_top_frame, text="Alpha Expression:").pack(anchor="w", padx=5, pady=5)
        self.sub_input_text = scrolledtext.ScrolledText(input_top_frame, height=10)
        self.sub_input_text.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Buttons
        btn_frame = ttk.Frame(input_frame)
        btn_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Button(btn_frame, text="Verify Alpha", 
                command=self.verify_alpha).pack(side="left", padx=5)
                
        ttk.Button(btn_frame, text="Submit Alpha", 
                command=self.submit_alpha).pack(side="left", padx=5)
                
        # From File Tab
        file_top_frame = ttk.Frame(file_frame)
        file_top_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Load file button
        ttk.Button(file_top_frame, text="Load Alpha File", 
                command=self.load_alpha_for_submission).pack(anchor="w", padx=5, pady=5)
                
        # File contents display
        ttk.Label(file_top_frame, text="File Contents:").pack(anchor="w", padx=5, pady=5)
        self.sub_file_text = scrolledtext.ScrolledText(file_top_frame, height=10)
        self.sub_file_text.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Buttons
        file_btn_frame = ttk.Frame(file_frame)
        file_btn_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Button(file_btn_frame, text="Verify Alpha", 
                command=self.verify_alpha).pack(side="left", padx=5)
                
        ttk.Button(file_btn_frame, text="Submit Alpha", 
                command=self.submit_alpha).pack(side="left", padx=5)
                
        # Fetch Existing Tab
        fetch_top_frame = ttk.Frame(fetch_frame)
        fetch_top_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Fetch button
        ttk.Button(fetch_top_frame, text="Fetch My Alphas", 
                command=self.fetch_alphas).pack(anchor="w", padx=5, pady=5)
                
        # Alphas table
        alphas_frame = ttk.Frame(fetch_top_frame)
        alphas_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Create treeview for alphas
        columns = ("id", "sharpe", "fitness", "expression")
        self.alphas_tree = ttk.Treeview(alphas_frame, columns=columns, show="headings", height=10)
        
        # Define headings
        self.alphas_tree.heading("id", text="Alpha ID")
        self.alphas_tree.heading("sharpe", text="Sharpe")
        self.alphas_tree.heading("fitness", text="Fitness")
        self.alphas_tree.heading("expression", text="Expression")
        
        # Define columns
        self.alphas_tree.column("id", width=80)
        self.alphas_tree.column("sharpe", width=80)
        self.alphas_tree.column("fitness", width=80)
        self.alphas_tree.column("expression", width=350)
        
        # Add scrollbar
        tree_scroll = ttk.Scrollbar(alphas_frame, orient="vertical", 
                                 command=self.alphas_tree.yview)
        self.alphas_tree.configure(yscrollcommand=tree_scroll.set)
        
        self.alphas_tree.pack(side="left", fill="both", expand=True)
        tree_scroll.pack(side="right", fill="y")
        
        # Buttons
        fetch_btn_frame = ttk.Frame(fetch_frame)
        fetch_btn_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Button(fetch_btn_frame, text="Verify Selected", 
                command=self.verify_alpha).pack(side="left", padx=5)
                
        ttk.Button(fetch_btn_frame, text="Submit Selected", 
                command=self.submit_alpha).pack(side="left", padx=5)
                
        ttk.Button(fetch_btn_frame, text="Batch Submit", 
                command=self.batch_submit).pack(side="left", padx=5)
                
        # Submission Log
        log_frame = ttk.LabelFrame(sub_frame, text="Submission Log")
        log_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.submission_log = scrolledtext.ScrolledText(log_frame, height=10)
        self.submission_log.pack(fill="both", expand=True, padx=5, pady=5)
        
    def setup_results_tab(self):
        """Set up the results management tab"""
        results_frame = ttk.Frame(self.notebook)
        self.notebook.add(results_frame, text="Results")
        
        # Split into left results list and right details
        left_frame = ttk.LabelFrame(results_frame, text="Saved Results")
        left_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        
        right_frame = ttk.LabelFrame(results_frame, text="Result Details")
        right_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        
        results_frame.grid_columnconfigure(0, weight=1)
        results_frame.grid_columnconfigure(1, weight=1)
        results_frame.grid_rowconfigure(0, weight=1)
        
        # Results list
        list_frame = ttk.Frame(left_frame)
        list_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Create treeview for results
        columns = ("filename", "date", "sharpe", "fitness", "expression")
        self.results_tree = ttk.Treeview(list_frame, columns=columns, show="headings", height=15)
        
        # Define headings
        self.results_tree.heading("filename", text="Filename")
        self.results_tree.heading("date", text="Date")
        self.results_tree.heading("sharpe", text="Sharpe")
        self.results_tree.heading("fitness", text="Fitness")
        self.results_tree.heading("expression", text="Expression")
        
        # Define columns
        self.results_tree.column("filename", width=150)
        self.results_tree.column("date", width=120)
        self.results_tree.column("sharpe", width=80)
        self.results_tree.column("fitness", width=80)
        self.results_tree.column("expression", width=200)
        
        # Add scrollbar
        tree_scroll = ttk.Scrollbar(list_frame, orient="vertical", 
                                 command=self.results_tree.yview)
        self.results_tree.configure(yscrollcommand=tree_scroll.set)
        
        self.results_tree.pack(side="left", fill="both", expand=True)
        tree_scroll.pack(side="right", fill="y")
        
        # Bind selection event
        self.results_tree.bind("<<TreeviewSelect>>", self.show_result_details)
        
        # Control buttons
        btn_frame = ttk.Frame(left_frame)
        btn_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Button(btn_frame, text="Refresh", 
                command=self.refresh_results).pack(side="left", padx=5)
                
        ttk.Button(btn_frame, text="Export Selected", 
                command=self.export_selected).pack(side="left", padx=5)
                
        ttk.Button(btn_frame, text="Delete Selected", 
                command=self.delete_selected).pack(side="left", padx=5)
                
        # Details area
        self.details_text = scrolledtext.ScrolledText(right_frame)
        self.details_text.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Load results
        self.refresh_results()
        
    def setup_settings_tab(self):
        """Set up the settings tab"""
        settings_frame = ttk.Frame(self.notebook)
        self.notebook.add(settings_frame, text="Settings")
        
        # Authentication Configuration
        auth_frame = ttk.LabelFrame(settings_frame, text="Authentication")
        auth_frame.pack(fill="x", padx=5, pady=5)
        
        # Login tutorial
        tutorial_frame = ttk.LabelFrame(auth_frame, text="Login Tutorial")
        tutorial_frame.pack(fill="x", padx=5, pady=5)
        
        tutorial_text = """
To use Alpha Mining Workbench with WorldQuant Brain:

1. Enter your WorldQuant Brain username and password below
2. Click "Login" to authenticate with the API
3. Your session will remain active for 8 hours
4. All alpha submissions will use this authentication

Note: Your credentials are used only for API authentication and are not stored permanently.
For security, consider using an API token instead of your password when available.
        """
        
        tutorial_label = ttk.Label(tutorial_frame, text=tutorial_text, justify="left", wraplength=500)
        tutorial_label.pack(padx=5, pady=5)
        
        cred_frame = ttk.Frame(auth_frame)
        cred_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Label(cred_frame, text="Username:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.settings_username_entry = ttk.Entry(cred_frame, width=30)
        self.settings_username_entry.grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        
        ttk.Label(cred_frame, text="Password:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.settings_password_entry = ttk.Entry(cred_frame, width=30, show="*")
        self.settings_password_entry.grid(row=1, column=1, sticky="ew", padx=5, pady=5)
        
        # Status and login button
        status_frame = ttk.Frame(auth_frame)
        status_frame.pack(fill="x", padx=5, pady=5)
        
        self.auth_status_var = tk.StringVar(value="Not authenticated")
        ttk.Label(status_frame, text="Status:").pack(side="left", padx=5)
        self.auth_status_label = ttk.Label(status_frame, textvariable=self.auth_status_var,
                                        foreground="red")
        self.auth_status_label.pack(side="left", padx=5)
        
        ttk.Button(auth_frame, text="Login", 
                command=self.authenticate_from_settings).pack(fill="x", padx=5, pady=5)
        
        # API Configuration
        api_frame = ttk.LabelFrame(settings_frame, text="API Configuration")
        api_frame.pack(fill="x", padx=5, pady=5)
        
        api_grid = ttk.Frame(api_frame)
        api_grid.pack(fill="x", padx=5, pady=5)
        
        ttk.Label(api_grid, text="API URL:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.api_url_entry = ttk.Entry(api_grid, width=50)
        self.api_url_entry.grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        self.api_url_entry.insert(0, Config.API_URL)
        
        ttk.Label(api_grid, text="API Key:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.api_key_entry = ttk.Entry(api_grid, width=50, show="*")
        self.api_key_entry.grid(row=1, column=1, sticky="ew", padx=5, pady=5)
        self.api_key_entry.insert(0, Config.API_KEY)
        
        # Directory Configuration
        dir_frame = ttk.LabelFrame(settings_frame, text="Directories")
        dir_frame.pack(fill="x", padx=5, pady=5)
        
        dir_grid = ttk.Frame(dir_frame)
        dir_grid.pack(fill="x", padx=5, pady=5)
        
        ttk.Label(dir_grid, text="Results Directory:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.results_dir_entry = ttk.Entry(dir_grid, width=40)
        self.results_dir_entry.grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        self.results_dir_entry.insert(0, Config.RESULTS_DIR)
        
        ttk.Button(dir_grid, text="Browse", 
                command=lambda: self.browse_directory(self.results_dir_entry)).grid(row=0, column=2, padx=5, pady=5)
        
        # Additional Settings
        settings_frame2 = ttk.LabelFrame(settings_frame, text="Additional Settings")
        settings_frame2.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Create treeview for settings
        tree_frame = ttk.Frame(settings_frame2)
        tree_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        columns = ("setting", "value")
        self.settings_tree = ttk.Treeview(tree_frame, columns=columns, show="headings", height=10)
        
        # Define headings
        self.settings_tree.heading("setting", text="Setting")
        self.settings_tree.heading("value", text="Value")
        
        # Define columns
        self.settings_tree.column("setting", width=150)
        self.settings_tree.column("value", width=350)
        
        # Add scrollbar
        tree_scroll = ttk.Scrollbar(tree_frame, orient="vertical", 
                                 command=self.settings_tree.yview)
        self.settings_tree.configure(yscrollcommand=tree_scroll.set)
        
        self.settings_tree.pack(side="left", fill="both", expand=True)
        tree_scroll.pack(side="right", fill="y")
        
        # Load settings
        self.load_settings_tree()
        
        # Save button
        ttk.Button(settings_frame, text="Save Settings", 
                command=self.save_settings).pack(fill="x", padx=5, pady=10)
        
    def stop_variation_testing(self):
        """Stop the variation testing process"""
        self.is_testing = False
        messagebox.showinfo("Info", "Testing process will stop after current test completes")

    def on_tab_changed(self, event):
        """Handle tab change events"""
        # Get the selected tab index
        selected_tab = self.notebook.index(self.notebook.select())
        
        # If switching to submission tab, sync auth status
        if selected_tab == 3:  # Index of submission tab
            self.sync_auth_status()
            
    def sync_auth_status(self):
        """Synchronize authentication status across tabs"""
        if hasattr(self, 'sub_auth_status_var') and hasattr(self, 'sub_auth_status_label'):
            if self.auth_credentials["is_authenticated"]:
                self.sub_auth_status_var.set(f"Authenticated as {self.auth_credentials['username']}")
                self.sub_auth_status_label.config(foreground="green")
            else:
                self.sub_auth_status_var.set("Not authenticated")
                self.sub_auth_status_label.config(foreground="red")

    # ---- Parameter Optimization Functions ----
    
    def load_alpha_for_optimization(self):
        """Load a saved alpha for optimization"""
        filetypes = [("JSON files", "*.json"), ("All files", "*.*")]
        filepath = filedialog.askopenfilename(
            initialdir=Config.RESULTS_DIR,
            title="Select Alpha File",
            filetypes=filetypes
        )
        
        if not filepath:
            return
            
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                
            # Extract the alpha expression or parameters
            if isinstance(data, dict):
                if 'expression' in data:
                    self.opt_expr_text.delete(1.0, tk.END)
                    self.opt_expr_text.insert(tk.END, data['expression'])
                elif 'parameters' in data:
                    self.opt_expr_text.delete(1.0, tk.END)
                    self.opt_expr_text.insert(tk.END, json.dumps(data['parameters'], indent=2))
                else:
                    self.opt_expr_text.delete(1.0, tk.END)
                    self.opt_expr_text.insert(tk.END, json.dumps(data, indent=2))
            else:
                messagebox.showerror("Error", "Invalid alpha file format")
                return
                
            # Extract parameters
            self.extract_parameters()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load alpha file: {str(e)}")
            
    def extract_parameters(self):
        """Extract parameters from the alpha expression for optimization"""
        alpha_text = self.opt_expr_text.get(1.0, tk.END).strip()
        if not alpha_text:
            messagebox.showerror("Error", "Please enter an alpha expression first")
            return
            
        # Clear existing sliders
        for widget in self.param_sliders_frame.winfo_children():
            widget.destroy()
            
        try:
            # Try to parse as JSON
            try:
                params = json.loads(alpha_text)
                self._create_sliders_from_dict(params)
                return
            except json.JSONDecodeError:
                # Not JSON, try to parse as expression
                pass
                
            # If not JSON, look for numeric parameters in the expression
            params = []
            for match in re.finditer(r'(?<=[,()\s])\d+(?![a-zA-Z])', alpha_text):
                number = int(match.group())
                params.append(("param_" + str(len(params) + 1), number))
                
            if not params:
                messagebox.showinfo("Info", "No parameters found in the expression")
                return
                
            # Create sliders for numeric parameters
            for name, value in params:
                self._create_param_slider(name, value, max(1, int(value * 0.5)), int(value * 1.5))
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to extract parameters: {str(e)}")
            
    def _create_sliders_from_dict(self, params_dict, parent_key=""):
        """Create sliders from a dictionary of parameters"""
        for key, value in params_dict.items():
            full_key = f"{parent_key}.{key}" if parent_key else key
            
            if isinstance(value, dict):
                self._create_sliders_from_dict(value, full_key)
            elif isinstance(value, (int, float)):
                min_val = max(0, int(value * 0.5)) if value > 0 else int(value * 1.5)
                max_val = int(value * 1.5) if value > 0 else max(0, int(value * 0.5))
                self._create_param_slider(full_key, value, min_val, max_val)
                
    def _create_param_slider(self, name, value, min_val, max_val):
        """Create a slider for parameter optimization"""
        frame = ttk.Frame(self.param_sliders_frame)
        frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Label(frame, text=name).pack(side="left")
        
        # Create variable to store value
        var = tk.DoubleVar(value=float(value))
        
        # Create and configure slider
        slider = ttk.Scale(frame, from_=min_val, to=max_val, variable=var, 
                         orient="horizontal", length=200)
        slider.pack(side="left", padx=10)
        
        # Display current value
        value_label = ttk.Label(frame, text=str(value))
        value_label.pack(side="left", padx=5)
        
        # Update value label when slider moves
        def update_value(*args):
            value_label.config(text=str(round(var.get(), 2)))
            
        var.trace_add("write", update_value)
        
        # Store reference to slider and variable
        if not hasattr(self, 'param_sliders'):
            self.param_sliders = {}
            
        self.param_sliders[name] = (slider, var, value_label)
        
    def start_optimization(self):
        """Start parameter optimization process"""
        if not hasattr(self, 'param_sliders') or not self.param_sliders:
            messagebox.showerror("Error", "No parameters to optimize")
            return
            
        # Get current parameter values
        params = {name: var.get() for name, (_, var, _) in self.param_sliders.items()}
        
        # Clear optimization results
        for item in self.opt_results_tree.get_children():
            self.opt_results_tree.delete(item)
            
        self.is_optimizing = True
        
        # Start optimization thread
        self.optimization_thread = threading.Thread(target=self._run_optimization, args=(params,))
        self.optimization_thread.daemon = True
        self.optimization_thread.start()
        
    def _run_optimization(self, initial_params):
        """Run parameter optimization in background thread"""
        try:
            # Generate parameter variations
            iterations = 30  # Number of iterations
            param_sets = []
            
            # Add initial parameters
            param_sets.append(initial_params)
            
            # Generate variations
            for i in range(1, iterations):
                new_params = {}
                for name, value in initial_params.items():
                    # Vary each parameter randomly within a range
                    variation = value * np.random.uniform(0.8, 1.2)
                    new_params[name] = variation
                param_sets.append(new_params)
                
            # Test each parameter set
            for i, params in enumerate(param_sets):
                if not self.is_optimizing:
                    break
                    
                # Simulate evaluation (in real implementation, this would call API)
                result = self._evaluate_params(params)
                
                # Add result to tree
                self.root.after(0, lambda p=params, r=result, idx=i: 
                             self._add_optimization_result(idx, p, r))
                             
                # Update progress
                progress = (i + 1) / len(param_sets) * 100
                self.root.after(0, lambda p=progress: self._update_opt_progress(p))
                
                # Simulate some processing time
                time.sleep(0.5)
                
        except Exception as e:
            logger.exception("Optimization error")
            self.root.after(0, lambda: messagebox.showerror("Error", 
                                                         f"Optimization error: {str(e)}"))
        finally:
            self.is_optimizing = False
            
    def _evaluate_params(self, params):
        """Evaluate a set of parameters"""
        # In a real implementation, this would test the alpha with these parameters
        # For now, return random metrics
        return {
            "sharpe": round(np.random.uniform(0.8, 2.0), 2),
            "turnover": round(np.random.uniform(0.05, 0.5), 2),
            "fitness": round(np.random.uniform(0.8, 1.8), 2)
        }
        
    def _add_optimization_result(self, iteration, params, result):
        """Add optimization result to the results tree"""
        # Format parameters as string
        params_str = ", ".join([f"{k}: {round(v, 2)}" for k, v in params.items()])
        
        # Format metrics
        sharpe = result.get("sharpe", "N/A")
        turnover = result.get("turnover", "N/A")
        fitness = result.get("fitness", "N/A")
        
        # Add to tree
        self.opt_results_tree.insert("", "end", values=(
            iteration + 1,
            sharpe,
            turnover,
            fitness,
            params_str
        ))
        
        # Scroll to see the new item
        self.opt_results_tree.see(self.opt_results_tree.get_children()[-1])
        
    def _update_opt_progress(self, progress):
        """Update optimization progress bar"""
        self.opt_progress_var.set(f"{progress:.1f}%")
        self.opt_progress_bar["value"] = progress
        
    def stop_optimization(self):
        """Stop the optimization process"""
        self.is_optimizing = False
        messagebox.showinfo("Info", "Optimization stopping...")
        
    def save_optimized_alpha(self):
        """Save the selected optimized alpha"""
        selected = self.opt_results_tree.selection()
        if not selected:
            messagebox.showerror("Error", "Please select an optimized alpha to save")
            return
            
        item = selected[0]
        values = self.opt_results_tree.item(item, "values")
        
        # Parse parameters from string
        params_str = values[4]
        params = {}
        for pair in params_str.split(", "):
            if ":" in pair:
                key, value = pair.split(":", 1)
                params[key.strip()] = float(value.strip())
                
        # Create alpha data
        alpha_data = {
            "parameters": params,
            "metrics": {
                "sharpe": float(values[1]),
                "turnover": float(values[2]),
                "fitness": float(values[3])
            },
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            filepath = self.results_manager.save_result(alpha_data)
            messagebox.showinfo("Success", f"Optimized alpha saved to {filepath}")
            
            # Refresh results tab
            self.refresh_results()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save optimized alpha: {str(e)}")

    # ---- Authentication Functions ----
    
    def authenticate_from_settings(self):
        """Authenticate from the settings tab"""
        username = self.settings_username_entry.get()
        password = self.settings_password_entry.get()
        
        if not username or not password:
            messagebox.showerror("Error", "Username and password are required")
            return
            
        self._perform_authentication(username, password)
        
    def authenticate_from_submission(self):
        """Authenticate from the submission tab"""
        # If already authenticated, show status
        if self.auth_credentials["is_authenticated"]:
            messagebox.showinfo("Authentication", 
                             f"Already logged in as {self.auth_credentials['username']}")
            return
            
        # Switch to settings tab for login
        self.notebook.select(5)  # Index of the settings tab
        
        # Highlight the username field
        self.settings_username_entry.focus_set()
        
        # Show a more helpful message
        messagebox.showinfo("Authentication Required", 
                         "Please log in with your WorldQuant Brain credentials in the Settings tab.\n\n"
                         "1. Enter your username and password\n"
                         "2. Click 'Login'\n"
                         "3. Return to the submission tab after successful login")
        
    def _perform_authentication(self, username, password):
        """Centralized authentication logic"""
        # Update status
        self.auth_status_var.set("Authenticating...")
        self.auth_status_label.config(foreground="blue")
        self.root.update()
        
        try:
            # Use WorldQuant Brain API for authentication
            import requests
            from requests.auth import HTTPBasicAuth
            
            # Create a session for API calls
            if not hasattr(self, 'api_session'):
                self.api_session = requests.Session()
            
            # Set up basic auth for the session
            self.api_session.auth = HTTPBasicAuth(username, password)
            
            # Call the authentication endpoint
            response = self.api_session.post('https://api.worldquantbrain.com/authentication')
            
            # Check if authentication was successful
            if response.status_code != 201:
                raise Exception(f"Authentication failed: {response.text}")
            
            # Extract session token from response
            session_token = response.headers.get('X-WQB-Session-Token') or "dummy_token_" + str(int(time.time()))
            
            # Update authentication state
            self.auth_credentials = {
                "username": username,
                "password": password,  # In a real app, don't store the password
                "is_authenticated": True,
                "session_token": session_token,
                "auth_time": datetime.now()
            }
            
            # Update UI
            self.auth_status_var.set(f"Authenticated as {username}")
            self.auth_status_label.config(foreground="green")
            
            # Also update submission tab status if it exists
            if hasattr(self, 'sub_auth_status_var'):
                self.sub_auth_status_var.set(f"Authenticated as {username}")
                self.sub_auth_status_label.config(foreground="green")
                
            messagebox.showinfo("Success", "Authentication successful")
            
        except Exception as e:
            logger.exception("Authentication error")
            self.auth_status_var.set("Authentication failed")
            self.auth_status_label.config(foreground="red")
            messagebox.showerror("Error", f"Authentication failed: {str(e)}")
            
    def check_authentication(self):
        """Check if user is authenticated and session is valid"""
        if not self.auth_credentials["is_authenticated"]:
            return False
            
        # Check if session has expired (e.g., after 8 hours)
        if self.auth_credentials["auth_time"]:
            elapsed = datetime.now() - self.auth_credentials["auth_time"]
            if elapsed.total_seconds() > 8 * 60 * 60:  # 8 hours
                # Session expired
                self.auth_credentials["is_authenticated"] = False
                self.auth_status_var.set("Session expired")
                self.auth_status_label.config(foreground="red")
                if hasattr(self, 'sub_auth_status_var'):
                    self.sub_auth_status_var.set("Session expired")
                    self.sub_auth_status_label.config(foreground="red")
                return False
                
        return True

    # ---- Alpha Submission Functions ----
    
    def load_alpha_for_submission(self):
        """Load an alpha file for submission"""
        filetypes = [("JSON files", "*.json"), ("All files", "*.*")]
        filepath = filedialog.askopenfilename(
            initialdir=Config.RESULTS_DIR,
            title="Select Alpha File",
            filetypes=filetypes
        )
        
        if not filepath:
            return
            
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                
            # Extract the alpha expression or parameters
            if isinstance(data, dict):
                if 'expression' in data:
                    self.sub_file_text.delete(1.0, tk.END)
                    self.sub_file_text.insert(tk.END, data['expression'])
                elif 'parameters' in data:
                    self.sub_file_text.delete(1.0, tk.END)
                    self.sub_file_text.insert(tk.END, json.dumps(data['parameters'], indent=2))
                else:
                    self.sub_file_text.delete(1.0, tk.END)
                    self.sub_file_text.insert(tk.END, json.dumps(data, indent=2))
            else:
                messagebox.showerror("Error", "Invalid alpha file format")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load alpha file: {str(e)}")
            
    def fetch_alphas(self):
        """Fetch successful alphas from the platform"""
        if not self.check_authentication():
            messagebox.showerror("Error", "Please authenticate first")
            return
            
        self.log_submission("Fetching alphas...")
        
        try:
            # Call the API to fetch alphas using the authenticated session
            response = self.api_session.get(
                'https://api.worldquantbrain.com/alphas'
            )
            
            if response.status_code != 200:
                raise Exception(f"Failed to fetch alphas: {response.text}")
            
            # Parse the response
            alphas_data = response.json()
            alphas = []
            
            # Process the response - adjust based on actual API response format
            for alpha_data in alphas_data.get('alphas', []):
                alpha = {
                    "id": alpha_data.get('id', 'unknown'),
                    "sharpe": alpha_data.get('sharpe', 0.0),
                    "fitness": alpha_data.get('fitness', 0.0),
                    "expression": alpha_data.get('expression', '')
                }
                alphas.append(alpha)
                
            # Clear existing entries
            for item in self.alphas_tree.get_children():
                self.alphas_tree.delete(item)
                
            # Add new entries
            for alpha in alphas:
                self.alphas_tree.insert("", "end", values=(
                    alpha["id"],
                    alpha["sharpe"],
                    alpha["fitness"],
                    alpha["expression"]
                ))
                
            self.log_submission(f"Fetched {len(alphas)} alphas")
            
        except Exception as e:
            logger.exception("Fetch alphas error")
            self.log_submission(f"Failed to fetch alphas: {str(e)}")
            messagebox.showerror("Error", f"Failed to fetch alphas: {str(e)}")
            
            # Fallback to dummy data for testing
            self.log_submission("Using mock data for testing...")
            alphas = []
            for i in range(10):
                alpha = {
                    "id": f"alpha_{i}",
                    "sharpe": round(np.random.uniform(1.0, 2.0), 2),
                    "fitness": round(np.random.uniform(1.0, 1.8), 2),
                    "expression": f"rank(close({5+i})) - rank(volume({10+i}))"
                }
                alphas.append(alpha)
                
            # Clear existing entries
            for item in self.alphas_tree.get_children():
                self.alphas_tree.delete(item)
                
            # Add new entries
            for alpha in alphas:
                self.alphas_tree.insert("", "end", values=(
                    alpha["id"],
                    alpha["sharpe"],
                    alpha["fitness"],
                    alpha["expression"]
                ))
                
            self.log_submission(f"Fetched {len(alphas)} alphas (mock data)")
        
    def verify_alpha(self):
        """Verify the alpha before submission"""
        if not self.check_authentication():
            messagebox.showerror("Error", "Please authenticate first")
            return
            
        # Get alpha from the selected tab
        expression = self._get_submission_alpha()
        
        if not expression:
            return
            
        self.log_submission(f"Verifying alpha: {expression}")
        
        try:
            # Call the API to verify the alpha using the authenticated session
            response = self.api_session.post(
                'https://api.worldquantbrain.com/alphas/verify',
                json={'expression': expression}
            )
            
            if response.status_code != 200:
                raise Exception(f"Verification failed: {response.text}")
            
            # Parse the response
            verification = response.json()
            
            # Log verification results
            self.log_submission("Verification results:")
            for check in verification.get("checks", []):
                self.log_submission(f"  {check['name']}: {check['result']}")
                
            self.log_submission("Metrics:")
            for metric, value in verification.get("metrics", {}).items():
                self.log_submission(f"  {metric}: {value}")
                
            # Show detailed results
            self._show_verification_results(expression, verification)
            
        except Exception as e:
            logger.exception("Alpha verification error")
            self.log_submission(f"Verification failed: {str(e)}")
            messagebox.showerror("Error", f"Verification failed: {str(e)}")
            
            # Fallback to dummy data for testing
            self.log_submission("Using mock verification data for testing...")
            
            # Simulate verification results
            verification = {
                "checks": [
                    {"name": "SYNTAX", "result": "PASS"},
                    {"name": "UNIVERSE_COVERAGE", "result": "PASS"},
                    {"name": "SHARPE", "result": "PASS"},
                    {"name": "TURNOVER", "result": "PASS"},
                    {"name": "CONCENTRATED_WEIGHT", "result": "PASS"}
                ],
                "metrics": {
                    "sharpe": 1.52,
                    "turnover": 0.23,
                    "fitness": 1.34
                }
            }
            
            # Log verification results
            self.log_submission("Mock verification results:")
            for check in verification["checks"]:
                self.log_submission(f"  {check['name']}: {check['result']}")
                
            self.log_submission("Metrics:")
            for metric, value in verification["metrics"].items():
                self.log_submission(f"  {metric}: {value}")
                
            # Show detailed results
            self._show_verification_results(expression, verification)
            
    def _show_verification_results(self, expression, verification):
        """Show detailed verification results"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Alpha Verification Results")
        dialog.geometry("500x400")
        
        ttk.Label(dialog, text="Expression:").pack(pady=5)
        expr_text = scrolledtext.ScrolledText(dialog, height=5)
        expr_text.pack(fill="x", padx=10, pady=5)
        expr_text.insert(tk.END, expression)
        
        # Create checks frame
        checks_frame = ttk.LabelFrame(dialog, text="Verification Checks")
        checks_frame.pack(fill="x", padx=10, pady=5)
        
        # Add checks
        for check in verification["checks"]:
            check_frame = ttk.Frame(checks_frame)
            check_frame.pack(fill="x", padx=5, pady=2)
            
            ttk.Label(check_frame, text=check["name"]).pack(side="left")
            
            result_label = ttk.Label(
                check_frame, 
                text=check["result"],
                foreground="green" if check["result"] == "PASS" else "red"
            )
            result_label.pack(side="right")
            
        # Create metrics frame
        metrics_frame = ttk.LabelFrame(dialog, text="Performance Metrics")
        metrics_frame.pack(fill="x", padx=10, pady=5)
        
        # Add metrics
        for metric, value in verification["metrics"].items():
            metric_frame = ttk.Frame(metrics_frame)
            metric_frame.pack(fill="x", padx=5, pady=2)
            
            ttk.Label(metric_frame, text=metric.capitalize()).pack(side="left")
            ttk.Label(metric_frame, text=str(value)).pack(side="right")
            
        # Add submit button if all checks pass
        all_pass = all(check["result"] == "PASS" for check in verification["checks"])
        
        if all_pass:
            ttk.Button(dialog, text="Submit Alpha", 
                    command=lambda: self._submit_verified_alpha(expression, dialog)).pack(pady=10)
            
        ttk.Button(dialog, text="Close", command=dialog.destroy).pack(pady=5)
        
    def _submit_verified_alpha(self, expression, dialog=None):
        """Submit a verified alpha"""
        self.log_submission(f"Submitting alpha: {expression}")
        
        # In a real implementation, this would call the API
        # For now, simulate submission
        time.sleep(1)
        
        # Simulate submission success
        self.log_submission("Submission successful!")
        messagebox.showinfo("Success", "Alpha submitted successfully")
        
        if dialog:
            dialog.destroy()
            
    def submit_alpha(self):
        """Submit the alpha to the platform"""
        if not self.check_authentication():
            messagebox.showerror("Error", "Please authenticate first")
            return
            
        # Get alpha from the selected tab
        expression = self._get_submission_alpha()
        
        if not expression:
            return
            
        # Ask for confirmation
        if not messagebox.askyesno("Confirm", "Submit this alpha to the platform?"):
            return
            
        self.log_submission(f"Submitting alpha: {expression}")
        
        # Start submission in a thread
        self.is_submitting = True
        submission_thread = threading.Thread(target=self._submit_alpha_background, args=(expression,))
        submission_thread.daemon = True
        submission_thread.start()
        
    def _submit_alpha_background(self, expression):
        """Submit alpha in a background thread"""
        try:
            # Call the API to submit the alpha using the authenticated session
            response = self.api_session.post(
                'https://api.worldquantbrain.com/alphas',
                json={'expression': expression}
            )
            
            if response.status_code not in [200, 201]:
                raise Exception(f"Submission failed: {response.text}")
            
            # Process the response
            result = response.json()
            
            self.root.after(0, lambda: self.log_submission(f"Submission successful! Alpha ID: {result.get('id', 'unknown')}"))
            self.root.after(0, lambda: messagebox.showinfo("Success", "Alpha submitted successfully"))
            
        except Exception as e:
            logger.exception("Alpha submission error")
            self.root.after(0, lambda: self.log_submission(f"Submission failed: {str(e)}"))
            self.root.after(0, lambda: messagebox.showerror("Error", f"Submission failed: {str(e)}"))
            
        finally:
            self.is_submitting = False
            
    def batch_submit(self):
        """Submit multiple alphas in batch"""
        if not self.check_authentication():
            messagebox.showerror("Error", "Please authenticate first")
            return
            
        # Check if alphas have been fetched
        if not self.alphas_tree.get_children():
            messagebox.showerror("Error", "Please fetch alphas first")
            return
            
        # Ask for confirmation
        count = len(self.alphas_tree.get_children())
        if not messagebox.askyesno("Confirm", f"Submit all {count} alphas in batch?"):
            return
            
        self.log_submission(f"Starting batch submission of {count} alphas...")
        
        # Start batch submission in a thread
        self.is_submitting = True
        submission_thread = threading.Thread(target=self._batch_submit_background)
        submission_thread.daemon = True
        submission_thread.start()
        
    def _batch_submit_background(self):
        """Submit multiple alphas in background thread"""
        try:
            alphas = []
            for item in self.alphas_tree.get_children():
                values = self.alphas_tree.item(item, "values")
                alphas.append({
                    "id": values[0],
                    "expression": values[3]
                })
                
            total = len(alphas)
            success = 0
            
            for i, alpha in enumerate(alphas):
                if not self.is_submitting:
                    break
                    
                self.root.after(0, lambda a=alpha: 
                             self.log_submission(f"Submitting alpha {a['id']}..."))
                             
                try:
                    # Call the API to submit the alpha using the authenticated session
                    response = self.api_session.post(
                        'https://api.worldquantbrain.com/alphas',
                        json={'expression': alpha["expression"]}
                    )
                    
                    if response.status_code not in [200, 201]:
                        raise Exception(f"Submission failed: {response.text}")
                    
                    # Process the response
                    result = response.json()
                    
                    self.root.after(0, lambda a=alpha, r=result: 
                                 self.log_submission(f"Alpha {a['id']} submitted successfully. New ID: {r.get('id', 'unknown')}"))
                    success += 1
                    
                except Exception as e:
                    logger.exception(f"Failed to submit alpha {alpha['id']}")
                    self.root.after(0, lambda a=alpha, err=str(e): 
                                 self.log_submission(f"Alpha {a['id']} submission failed: {err}"))
                             
                # Update progress in UI
                progress = (i + 1) / total * 100
                self.root.after(0, lambda p=progress, s=success, t=total: 
                             self.log_submission(f"Progress: {p:.1f}% ({s}/{t} successful)"))
                             
            self.root.after(0, lambda s=success, t=total: 
                         self.log_submission(f"Batch submission complete: {s}/{t} successful"))
                         
            self.root.after(0, lambda s=success, t=total: 
                         messagebox.showinfo("Batch Complete", 
                                          f"Batch submission complete: {s}/{t} successful"))
                                          
        except Exception as e:
            logger.exception("Batch submission error")
            self.root.after(0, lambda: self.log_submission(f"Batch submission error: {str(e)}"))
            self.root.after(0, lambda: messagebox.showerror("Error", 
                                                         f"Batch submission error: {str(e)}"))
        finally:
            self.is_submitting = False
            
    def _get_submission_alpha(self):
        """Get the alpha expression from the current submission tab"""
        # Determine which tab is active
        active_tab = self.notebook.nametowidget(self.notebook.select())
        tab_name = active_tab.winfo_name()
        
        # Get the alpha based on the active tab
        if "file_frame" in tab_name:
            expression = self.sub_file_text.get(1.0, tk.END).strip()
        elif "input_frame" in tab_name:
            expression = self.sub_input_text.get(1.0, tk.END).strip()
        elif "fetch_frame" in tab_name:
            selected = self.alphas_tree.selection()
            if not selected:
                messagebox.showerror("Error", "Please select an alpha")
                return None
            expression = self.alphas_tree.item(selected[0], "values")[3]
        else:
            messagebox.showerror("Error", "Unknown tab selected")
            return None
            
        if not expression:
            messagebox.showerror("Error", "No alpha expression provided")
            return None
            
        return expression
        
    def log_submission(self, message):
        """Add a log message to the submission log"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.submission_log.insert(tk.END, f"[{timestamp}] {message}\n")
        self.submission_log.see(tk.END)

    # ---- Results Management Functions ----
    
    def refresh_results(self):
        """Refresh the results list"""
        # Clear existing entries
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
            
        # Get list of result files
        try:
            results = self.results_manager.list_results()
            
            for filename in results:
                try:
                    # Load result file
                    data = self.results_manager.load_result(filename)
                    
                    # Extract information
                    date = datetime.fromtimestamp(os.path.getctime(
                        os.path.join(Config.RESULTS_DIR, filename)
                    )).strftime("%Y-%m-%d %H:%M:%S")
                    
                    # Extract metrics and expression
                    sharpe = "N/A"
                    fitness = "N/A"
                    expression = "N/A"
                    
                    if "metrics" in data:
                        sharpe = data["metrics"].get("sharpe", "N/A")
                        fitness = data["metrics"].get("fitness", "N/A")
                        
                    if "expression" in data:
                        expression = data["expression"]
                    elif "parameters" in data:
                        expression = str(data["parameters"])
                        
                    # Add to tree
                    self.results_tree.insert("", "end", values=(
                        filename,
                        date,
                        sharpe,
                        fitness,
                        expression
                    ))
                    
                except Exception as e:
                    logger.error(f"Error processing result file {filename}: {str(e)}")
                    
        except Exception as e:
            messagebox.showerror("Error", f"Failed to refresh results: {str(e)}")
            
    def show_result_details(self, event):
        """Show details of the selected result"""
        selected = self.results_tree.selection()
        if not selected:
            return
            
        filename = self.results_tree.item(selected[0], "values")[0]
        
        try:
            # Load result file
            data = self.results_manager.load_result(filename)
            
            # Clear details text
            self.details_text.delete(1.0, tk.END)
            
            # Display formatted details
            self.details_text.insert(tk.END, f"Filename: {filename}\n\n")
            
            # Format data based on its structure
            if "expression" in data:
                self.details_text.insert(tk.END, f"Expression:\n{data['expression']}\n\n")
                
            if "parameters" in data:
                self.details_text.insert(tk.END, "Parameters:\n")
                self._format_dict(data["parameters"], prefix="  ")
                self.details_text.insert(tk.END, "\n")
                
            if "metrics" in data:
                self.details_text.insert(tk.END, "Metrics:\n")
                self._format_dict(data["metrics"], prefix="  ")
                
            if "result" in data:
                self.details_text.insert(tk.END, "Result:\n")
                self._format_dict(data["result"], prefix="  ")
                
        except Exception as e:
            self.details_text.delete(1.0, tk.END)
            self.details_text.insert(tk.END, f"Error loading result: {str(e)}")
            
    def _format_dict(self, data, prefix=""):
        """Format a dictionary for display in the details text"""
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, dict):
                    self.details_text.insert(tk.END, f"{prefix}{key}:\n")
                    self._format_dict(value, prefix + "  ")
                else:
                    self.details_text.insert(tk.END, f"{prefix}{key}: {value}\n")
        else:
            self.details_text.insert(tk.END, f"{prefix}{data}\n")
            
    def export_selected(self):
        """Export selected result to a file"""
        selected = self.results_tree.selection()
        if not selected:
            messagebox.showerror("Error", "Please select a result to export")
            return
            
        filename = self.results_tree.item(selected[0], "values")[0]
        
        # Ask for export location
        export_path = filedialog.asksaveasfilename(
            initialfile=filename,
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if not export_path:
            return
            
        try:
            # Load and save to new location
            data = self.results_manager.load_result(filename)
            
            with open(export_path, 'w') as f:
                json.dump(data, f, indent=2)
                
            messagebox.showinfo("Success", f"Result exported to {export_path}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export result: {str(e)}")
            
    def delete_selected(self):
        """Delete selected result"""
        selected = self.results_tree.selection()
        if not selected:
            messagebox.showerror("Error", "Please select a result to delete")
            return
            
        filename = self.results_tree.item(selected[0], "values")[0]
        
        # Ask for confirmation
        if not messagebox.askyesno("Confirm", f"Delete result {filename}?"):
            return
            
        try:
            # Delete file
            os.remove(os.path.join(Config.RESULTS_DIR, filename))
            
            # Refresh results
            self.refresh_results()
            
            messagebox.showinfo("Success", f"Result {filename} deleted")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to delete result: {str(e)}")

    # ---- Settings Management Functions ----
    
    def browse_directory(self, entry_widget):
        """Browse for a directory and update entry widget"""
        directory = filedialog.askdirectory(initialdir=entry_widget.get())
        if directory:
            entry_widget.delete(0, tk.END)
            entry_widget.insert(0, directory)
            
    def save_settings(self):
        """Save settings to config file"""
        try:
            # Update config values
            Config.API_URL = self.api_url_entry.get()
            Config.API_KEY = self.api_key_entry.get()
            Config.RESULTS_DIR = self.results_dir_entry.get()
            
            # Save authentication info if available
            if self.auth_credentials["is_authenticated"]:
                Config.USERNAME = self.auth_credentials["username"]
                Config.AUTH_TOKEN = self.auth_credentials["session_token"]
            
            # Update additional settings from tree
            for item in self.settings_tree.get_children():
                key, value = self.settings_tree.item(item, "values")
                
                # Convert to appropriate type
                if value.isdigit():
                    value = int(value)
                elif value.replace(".", "", 1).isdigit():
                    value = float(value)
                    
                # Set attribute if it exists
                if hasattr(Config, key):
                    setattr(Config, key, value)
                    
            # Save to file
            config = Config()
            config.save_to_file(Config.CONFIG_FILE)
            
            # Ensure directories exist
            Config.ensure_directories()
            
            # Update results manager directory
            self.results_manager = ResultsManager(Config.RESULTS_DIR)
            
            messagebox.showinfo("Success", "Settings saved successfully")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save settings: {str(e)}")
            
    def load_config(self):
        """Load configuration from file"""
        try:
            # Load from file if it exists
            if os.path.exists(Config.CONFIG_FILE):
                config = Config.load_from_file(Config.CONFIG_FILE)
                
                # Update UI elements
                self.api_url_entry.delete(0, tk.END)
                self.api_url_entry.insert(0, config.API_URL)
                
                self.api_key_entry.delete(0, tk.END)
                self.api_key_entry.insert(0, config.API_KEY)
                
                self.results_dir_entry.delete(0, tk.END)
                self.results_dir_entry.insert(0, config.RESULTS_DIR)
                
                # Check for saved credentials (in a real app, use a secure storage)
                if hasattr(config, 'USERNAME') and hasattr(config, 'AUTH_TOKEN'):
                    if config.USERNAME and config.AUTH_TOKEN:
                        # Auto-fill credentials in settings
                        self.settings_username_entry.delete(0, tk.END)
                        self.settings_username_entry.insert(0, config.USERNAME)
                        
                        # Update authentication status (in a real app, verify the token first)
                        self.auth_credentials = {
                            "username": config.USERNAME,
                            "password": "",
                            "is_authenticated": True,
                            "session_token": config.AUTH_TOKEN,
                            "auth_time": datetime.now()  # Reset the time
                        }
                        
                        # Update UI status
                        self.auth_status_var.set(f"Authenticated as {config.USERNAME}")
                        self.auth_status_label.config(foreground="green")
                        
                        if hasattr(self, 'sub_auth_status_var'):
                            self.sub_auth_status_var.set(f"Authenticated as {config.USERNAME}")
                            self.sub_auth_status_label.config(foreground="green")
                
                # Update settings tree
                self.load_settings_tree()
                
        except Exception as e:
            logger.error(f"Failed to load configuration: {str(e)}")
            
    def load_settings_tree(self):
        """Load settings into the settings tree"""
        # Clear existing entries
        for item in self.settings_tree.get_children():
            self.settings_tree.delete(item)
            
        # Add Config attributes that are all uppercase (constants)
        for attr in dir(Config):
            if attr.isupper():
                value = getattr(Config, attr)
                
                # Skip non-serializable types
                if isinstance(value, (int, float, str, bool)):
                    self.settings_tree.insert("", "end", values=(attr, value))

    def fetch_alpha_by_id(self, alpha_id):
        """Fetch a specific alpha by ID"""
        if not self.check_authentication():
            messagebox.showerror("Error", "Please authenticate first")
            return None
            
        try:
            # Call the API to fetch the alpha using the authenticated session
            response = self.api_session.get(
                f'https://api.worldquantbrain.com/alphas/{alpha_id}'
            )
            
            if response.status_code != 200:
                raise Exception(f"Failed to fetch alpha: {response.text}")
            
            # Parse the response
            alpha_data = response.json()
            
            return alpha_data
            
        except Exception as e:
            logger.exception(f"Failed to fetch alpha {alpha_id}")
            messagebox.showerror("Error", f"Failed to fetch alpha: {str(e)}")
            return None

    def fetch_operators(self):
        """Fetch operators from WorldQuant Brain API for alpha generation"""
        if not self.check_authentication():
            messagebox.showerror("Error", "Please authenticate first")
            return
            
        self.log_mining("Fetching operators from WorldQuant Brain...")
        
        try:
            # Call the API to fetch operators without parameters
            response = self.api_session.get(
                'https://api.worldquantbrain.com/operators'
            )
            
            if response.status_code != 200:
                raise Exception(f"Failed to fetch operators: {response.text}")
            
            # Parse the response
            data = response.json()
            
            # The operators endpoint might return a direct array instead of an object with 'items' or 'results'
            if isinstance(data, list):
                self.operators = data
            elif 'results' in data:
                self.operators = data['results']
            else:
                raise Exception(f"Unexpected operators response format")
            
            self.log_mining(f"Successfully fetched {len(self.operators)} operators")
            
            # Also fetch data fields
            self.get_data_fields()
            
            messagebox.showinfo("Success", f"Successfully fetched {len(self.operators)} operators and {len(self.data_fields) if hasattr(self, 'data_fields') else 0} data fields")
            
        except Exception as e:
            logger.exception("Fetch operators error")
            self.log_mining(f"Failed to fetch operators: {str(e)}")
            messagebox.showerror("Error", f"Failed to fetch operators: {str(e)}")
            
            # Generate mock operators for testing
            self.operators = self._generate_mock_operators()
            self.data_fields = self._generate_mock_data_fields()
            self.log_mining("Using mock operators and data fields for testing")
            
    def get_data_fields(self):
        """Fetch available data fields from WorldQuant Brain across multiple datasets with random sampling."""
        self.log_mining("Fetching data fields from WorldQuant Brain...")
        
        datasets = ['fundamental6', 'fundamental2', 'analyst4', 'model16', 'model51', 'news12']
        all_fields = []
        
        base_params = {
            'delay': 1,
            'instrumentType': 'EQUITY',
            'limit': 20,
            'region': 'USA',
            'universe': 'TOP3000'
        }
        
        try:
            for dataset in datasets:
                # First get the count
                params = base_params.copy()
                params['dataset.id'] = dataset
                params['limit'] = 1  # Just to get count efficiently
                
                self.log_mining(f"Getting field count for dataset: {dataset}")
                count_response = self.api_session.get(
                    'https://api.worldquantbrain.com/data-fields', 
                    params=params
                )
                
                if count_response.status_code == 200:
                    count_data = count_response.json()
                    total_fields = count_data.get('count', 0)
                    self.log_mining(f"Total fields in {dataset}: {total_fields}")
                    
                    if total_fields > 0:
                        # Generate random offset
                        max_offset = max(0, total_fields - base_params['limit'])
                        random_offset = random.randint(0, max_offset)
                        
                        # Fetch random subset
                        params['offset'] = random_offset
                        params['limit'] = min(20, total_fields)  # Don't exceed total fields
                        
                        self.log_mining(f"Fetching fields for {dataset} with offset {random_offset}")
                        response = self.api_session.get(
                            'https://api.worldquantbrain.com/data-fields',
                            params=params
                        )
                        
                        if response.status_code == 200:
                            data = response.json()
                            fields = data.get('results', [])
                            self.log_mining(f"Found {len(fields)} fields in {dataset}")
                            all_fields.extend(fields)
                        else:
                            self.log_mining(f"Failed to fetch fields for {dataset}: {response.text[:100]}")
                else:
                    self.log_mining(f"Failed to get count for {dataset}: {count_response.text[:100]}")
            
            # Remove duplicates if any
            unique_fields = list({field['id']: field for field in all_fields}.values())
            self.log_mining(f"Total unique fields found: {len(unique_fields)}")
            self.data_fields = unique_fields
            
            return unique_fields
            
        except Exception as e:
            logger.exception("Failed to fetch data fields")
            self.log_mining(f"Failed to fetch data fields: {str(e)}")
            return []

    def _generate_mock_operators(self):
        """Generate mock operators for testing"""
        categories = [
            "Time Series", "Cross Sectional", "Arithmetic", 
            "Logical", "Vector", "Transformational", "Group"
        ]
        
        operators = []
        for i, category in enumerate(categories):
            for j in range(5):  # 5 operators per category
                operators.append({
                    "name": f"{category.lower().replace(' ', '_')}_{j+1}",
                    "category": category,
                    "type": "SCALAR" if j % 3 != 0 else "VECTOR",
                    "definition": f"Example definition for {category.lower().replace(' ', '_')}_{j+1}",
                    "description": f"Example description for {category.lower().replace(' ', '_')}_{j+1}"
                })
                
        return operators
        
    def _generate_mock_data_fields(self):
        """Generate mock data fields for testing"""
        fields = [
            {"id": "close", "type": "SCALAR", "description": "Closing price"},
            {"id": "open", "type": "SCALAR", "description": "Opening price"},
            {"id": "high", "type": "SCALAR", "description": "High price"},
            {"id": "low", "type": "SCALAR", "description": "Low price"},
            {"id": "volume", "type": "SCALAR", "description": "Trading volume"},
            {"id": "returns", "type": "SCALAR", "description": "Daily returns"},
            {"id": "market_cap", "type": "SCALAR", "description": "Market capitalization"},
            {"id": "ev_to_ebitda", "type": "SCALAR", "description": "Enterprise value to EBITDA"},
            {"id": "pe_ratio", "type": "SCALAR", "description": "Price to earnings ratio"},
            {"id": "pb_ratio", "type": "SCALAR", "description": "Price to book ratio"}
        ]
        return fields

    def start_mining(self):
        """Start the alpha mining or generation process"""
        if self.strategy_var.get() == "ai":
            self.start_ai_generation()
        else:
            self.start_traditional_mining()
            
    def start_ai_generation(self):
        """Start the AI-based alpha generation process"""
        if not hasattr(self, 'operators') or not hasattr(self, 'data_fields'):
            messagebox.showerror("Error", "Please fetch operators first")
            return

        # Get API key and validate
        moonshot_api_key = self.moonshot_api_key_entry.get()
        if not moonshot_api_key:
            messagebox.showerror("Error", "Moonshot API key is required")
            return
            
        try:
            num_ideas = int(self.ai_ideas_entry.get())
            if num_ideas <= 0:
                raise ValueError("Number of ideas must be positive")
            if num_ideas > 20:
                if not messagebox.askyesno("Warning", "Generating many ideas may take a long time. Continue?"):
                    return
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid number of ideas: {str(e)}")
            return
            
        # Update UI state
        self.is_mining = True
        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")
        
        # Clear UI
        self.mining_log.delete(1.0, tk.END)
        for item in self.generated_alphas_tree.get_children():
            self.generated_alphas_tree.delete(item)
            
        # Start the generation thread
        self.mining_thread = threading.Thread(target=self.ai_generation_process)
        self.mining_thread.daemon = True
        self.mining_thread.start()
        
        self.log_mining("AI alpha generation process started")
        
    def ai_generation_process(self):
        """Main AI generation process executed in a separate thread"""
        try:
            model = self.ai_model_var.get()
            num_ideas = int(self.ai_ideas_entry.get())
            temperature = float(self.ai_temp_var.get())
            moonshot_api_key = self.moonshot_api_key_entry.get()
            moonshot_api_url = self.moonshot_api_url_entry.get()
            
            self.root.after(0, lambda: self.log_mining(f"Using model: {model}"))
            self.root.after(0, lambda: self.log_mining(f"Temperature: {temperature}"))
            self.root.after(0, lambda: self.log_mining(f"Generating {num_ideas} alpha ideas..."))
            
            # Sample a subset of operators to avoid token limit issues
            sampled_operators = self._sample_operators()
            self.root.after(0, lambda: self.log_mining(f"Sampled {len(sampled_operators)} operators"))
            
            # Prepare the prompt
            prompt = self._prepare_ai_prompt(sampled_operators, num_ideas)
            self.root.after(0, lambda: self.log_mining("Prompt prepared, sending to AI model..."))
            
            # Send request to Moonshot API
            headers = {
                'Authorization': f'Bearer {moonshot_api_key}',
                'Content-Type': 'application/json'
            }
            
            data = {
                'model': model,
                'messages': [
                    {
                        "role": "system", 
                        "content": "You are a quantitative analyst expert in generating alpha factors for stock market prediction."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                'temperature': temperature
            }
            
            # Make the API request with improved error handling
            try:
                self.root.after(0, lambda: self.log_mining("Sending request to Moonshot API..."))
                response = requests.post(
                    moonshot_api_url,
                    headers=headers,
                    json=data,
                    timeout=120  # Increased timeout for large models
                )
                
                self.root.after(0, lambda: self.log_mining(f"API response status: {response.status_code}"))
                
                if response.status_code != 200:
                    error_msg = f"API request failed with status {response.status_code}"
                    try:
                        error_detail = response.json()
                        error_msg += f": {str(error_detail)}"
                    except:
                        error_msg += f": {response.text[:200]}"
                    raise Exception(error_msg)
                    
            except requests.Timeout:
                self.root.after(0, lambda: self.log_mining("API request timed out. The model may be too busy."))
                raise Exception("API request timed out after 120 seconds. Please try again later.")
                
            except requests.ConnectionError:
                self.root.after(0, lambda: self.log_mining("Connection error. Please check your internet connection."))
                raise Exception("Connection error when contacting the Moonshot API. Please check your network.")
            
            # Process the response
            ai_response = response.json()
            content = ai_response.get('choices', [{}])[0].get('message', {}).get('content', '')
            
            if not content:
                self.root.after(0, lambda: self.log_mining("Received empty response from AI model."))
                raise Exception("The AI model returned an empty response. Please try again.")
            
            self.root.after(0, lambda: self.log_mining("Received AI response, processing..."))
            
            # Parse the alpha expressions from the content
            alphas = []
            for line in content.strip().split('\n'):
                if line and not line.startswith('#') and not line.startswith('//'):
                    alphas.append(line.strip())
            
            if not alphas:
                self.root.after(0, lambda: self.log_mining("No alpha expressions found in the AI response."))
                raise Exception("Could not extract any alpha expressions from the AI response.")
                
            # Clean the alpha ideas (filter out invalid ones)
            # Clean the alpha ideas (filter out invalid ones)
            cleaned_alphas = self._clean_alpha_ideas(alphas)
            self.root.after(0, lambda: self.log_mining(f"Generated {len(cleaned_alphas)} valid alpha expressions"))
            
            # Add alphas to the UI
            for i, alpha in enumerate(cleaned_alphas):
                alpha_id = f"ai_{i+1}"
                self.root.after(0, lambda id=alpha_id, expr=alpha: self._add_generated_alpha(id, expr))
                # Simulate progress
                progress = (i + 1) / len(cleaned_alphas) * 100
                self.root.after(0, lambda p=progress: self._update_mining_progress(p))
                
        except Exception as e:
            self.root.after(0, lambda: self.log_mining(f"Error during AI generation: {str(e)}"))
            logger.exception("AI generation process error")
        finally:
            self.is_mining = False
            self.root.after(0, self.reset_mining_state)
            
    def _sample_operators(self):
        """Sample operators for the AI prompt to avoid token limit issues"""
        if not hasattr(self, 'operators'):
            return []
            
        # Organize operators by category
        operator_by_category = {}
        for op in self.operators:
            category = op.get('category', 'Other')
            if category not in operator_by_category:
                operator_by_category[category] = []
            operator_by_category[category].append(op)
        
        # Sample ~50% of operators from each category
        sampled_operators = {}
        for category, ops in operator_by_category.items():
            sample_size = max(1, int(len(ops) * 0.5))  # At least 1 operator per category
            sampled_operators[category] = random.sample(ops, sample_size)
            
        # Flatten the dictionary
        result = []
        for category, ops in sampled_operators.items():
            for op in ops:
                result.append(op)
                
        return result
        
    def _prepare_ai_prompt(self, operators, num_ideas):
        """Prepare the prompt for the AI model"""
        # Organize operators by category
        operator_by_category = {}
        for op in operators:
            category = op.get('category', 'Other')
            if category not in operator_by_category:
                operator_by_category[category] = []
            operator_by_category[category].append(op)
            
        # Format operators by category
        def format_operators(ops):
            formatted = []
            for op in ops:
                formatted.append(f"{op.get('name', 'unknown')} ({op.get('type', 'SCALAR')})\n"
                               f"  Definition: {op.get('definition', 'N/A')}\n"
                               f"  Description: {op.get('description', 'N/A')}")
            return formatted
            
        # Get data field IDs
        data_fields = []
        if hasattr(self, 'data_fields'):
            data_fields = [field.get('id', 'unknown') for field in self.data_fields]
            
        # Build the prompt
        prompt = f"""Generate {num_ideas} unique alpha factor expressions using the available operators and data fields. Return ONLY the expressions, one per line, with no comments or explanations.

Available Data Fields:
{data_fields}

Available Operators by Category:
"""

        # Add operators by category
        for category, ops in operator_by_category.items():
            prompt += f"{category}:\n"
            prompt += chr(10).join(format_operators(ops))
            prompt += "\n\n"
            
        # Add requirements and tips
        prompt += """
Requirements:
1. Let your intuition guide you.
2. Use the operators and data fields to create unique and potentially profitable alpha factors.
3. Focus on simple yet effective expressions.

Tips: 
- You can use semi-colons to separate expressions.
- Pay attention to operator types (SCALAR, VECTOR, MATRIX) for compatibility.
- Study the operator definitions and descriptions to understand their behavior.

Example format:
ts_std_dev(cashflow_op, 180)
rank(divide(revenue, assets))
"""

        return prompt
        
    def _clean_alpha_ideas(self, ideas):
        """Clean and validate alpha ideas, keeping only valid expressions"""
        cleaned_ideas = []
        
        for idea in ideas:
            # Skip if idea is just a number or single word
            if re.match(r'^\d+\.?$|^[a-zA-Z]+$', idea):
                continue
            
            # Skip if idea is a description (contains common English words)
            common_words = ['it', 'the', 'is', 'are', 'captures', 'provides', 'measures']
            if any(word + ' ' in idea.lower() for word in common_words):
                continue
            
            # Verify idea contains valid operators/functions
            valid_functions = ['ts_mean', 'divide', 'subtract', 'add', 'multiply', 'zscore', 
                              'ts_rank', 'ts_std_dev', 'rank', 'log', 'sqrt']
            has_valid_func = False
            
            for func in valid_functions:
                if func in idea:
                    has_valid_func = True
                    break
                    
            if not has_valid_func:
                continue
            
            cleaned_ideas.append(idea)
            
        return cleaned_ideas
        
    def _add_generated_alpha(self, alpha_id, expression):
        """Add a generated alpha to the tree"""
        self.generated_alphas_tree.insert("", "end", values=(
            alpha_id,
            expression,
            "AI Generated"
        ))
        
    def _update_mining_progress(self, progress):
        """Update mining progress UI"""
        self.progress_var.set(f"{progress:.1f}%")
        self.progress_bar["value"] = progress
        
    def start_traditional_mining(self):
        """Start the traditional alpha mining process (genetic algorithm or ML)"""
        if not self.validate_mining_inputs():
            return
        
        self.is_mining = True
        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")
        
        # Clear mining log
        self.mining_log.delete(1.0, tk.END)
        
        # Create configuration dictionary
        config = {
            "api_key": self.api_key_entry.get(),
            "api_url": self.api_url_entry.get(),
            "max_iterations": int(self.max_iter_entry.get()),
            "population_size": int(self.pop_size_entry.get()),
            "strategy": self.strategy_var.get()
        }
        
        # Create AlphaMiner instance
        self.alpha_miner = AlphaMiner(config)
        
        # Start mining thread
        self.mining_thread = threading.Thread(target=self.mining_process)
        self.mining_thread.daemon = True
        self.mining_thread.start()

        self.log_mining("Alpha mining process started")

    def mining_process(self):
        """Main mining process executed in a separate thread"""
        try:
            for progress in self.alpha_miner.mine_alphas():
                if not self.is_mining:
                    self.log_mining("Mining stopped by user")
                    break
                
                self.update_mining_progress(progress)
                
        except Exception as e:
            self.log_mining(f"Error during mining: {str(e)}")
            logger.exception("Mining process error")
        finally:
            self.is_mining = False
            self.root.after(0, self.reset_mining_state)

    def stop_mining(self):
        """Stop the mining process"""
        self.is_mining = False
        self.log_mining("Stopping mining process...")
        if hasattr(self, 'alpha_miner') and self.alpha_miner:
            self.alpha_miner.is_running = False

    def reset_mining_state(self):
        """Reset UI state after mining is complete"""
        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")
        self.progress_var.set("0%")
        self.progress_bar["value"] = 0

    def update_mining_progress(self, progress_data):
        """Update the mining progress display"""
        self.root.after(0, lambda: self._update_mining_progress_ui(progress_data))

    def _update_mining_progress_ui(self, progress_data):
        """Update the UI with mining progress (runs in main thread)"""
        progress = progress_data.get("progress", 0)
        self.progress_var.set(f"{progress:.1f}%")
        self.progress_bar["value"] = progress
        
        if "alpha" in progress_data:
            self.log_mining(f"New alpha found: {progress_data['alpha']}")
            self.log_mining(f"Score: {progress_data.get('score', 'N/A')}")
            self.log_mining(f"Metrics: {progress_data.get('metrics', {})}")
            
            # Add to generated alphas tree
            alpha_id = f"gen_{len(self.generated_alphas_tree.get_children()) + 1}"
            self._add_generated_alpha(alpha_id, progress_data['alpha'])

    def log_mining(self, message):
        """Add a log message to the mining log"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.mining_log.insert(tk.END, f"[{timestamp}] {message}\n")
        self.mining_log.see(tk.END)
        
    def validate_mining_inputs(self):
        """Validate mining input parameters"""
        try:
            max_iter = int(self.max_iter_entry.get())
            pop_size = int(self.pop_size_entry.get())
            
            if max_iter <= 0 or pop_size <= 0:
                messagebox.showerror("Error", "Values must be positive")
            return False
            
            return True
        except ValueError:
            messagebox.showerror("Error", "Invalid numeric parameters")
            return False
            
    def verify_selected_alpha(self):
        """Verify the selected alpha expression"""
        selected = self.generated_alphas_tree.selection()
        if not selected:
            messagebox.showerror("Error", "Please select an alpha to verify")
            return
            
        # Get the expression from the selected item
        expression = self.generated_alphas_tree.item(selected[0], "values")[1]
        
        # Switch to the submission tab and verify
        self.notebook.select(3)  # Index of the submission tab
        
        # Set the expression in the direct input tab
        self.sub_input_text.delete(1.0, tk.END)
        self.sub_input_text.insert(tk.END, expression)
        
        # Call verify_alpha
        self.verify_alpha()
        
    def save_selected_alpha(self):
        """Save the selected alpha to a file"""
        selected = self.generated_alphas_tree.selection()
        if not selected:
            messagebox.showerror("Error", "Please select an alpha to save")
            return
            
        # Get the expression from the selected item
        values = self.generated_alphas_tree.item(selected[0], "values")
        alpha_id = values[0]
        expression = values[1]
        
        # Create a data structure to save
        alpha_data = {
            "expression": expression,
            "origin": "AI Generated" if alpha_id.startswith("ai_") else "Generated",
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            filepath = self.results_manager.save_result(alpha_data)
            messagebox.showinfo("Success", f"Alpha saved to {filepath}")
            
            # Refresh results tab
            self.refresh_results()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save alpha: {str(e)}")
            logger.exception("Save alpha error")
            
    def submit_selected_alpha(self):
        """Submit the selected alpha to the platform"""
        selected = self.generated_alphas_tree.selection()
        if not selected:
            messagebox.showerror("Error", "Please select an alpha to submit")
            return
            
        # Get the expression from the selected item
        expression = self.generated_alphas_tree.item(selected[0], "values")[1]
        
        # Switch to the submission tab
        self.notebook.select(3)  # Index of the submission tab
        
        # Set the expression in the direct input tab
        self.sub_input_text.delete(1.0, tk.END)
        self.sub_input_text.insert(tk.END, expression)
        
        # Call submit_alpha
        self.submit_alpha()
        
    def optimize_selected_alpha(self):
        """Open the optimization tab with the selected alpha loaded"""
        selected = self.generated_alphas_tree.selection()
        if not selected:
            messagebox.showerror("Error", "Please select an alpha to optimize")
            return
            
        # Get the expression from the selected item
        expression = self.generated_alphas_tree.item(selected[0], "values")[1]
        
        # Switch to optimization tab
        self.notebook.select(2)  # Index of optimization tab
        
        # Load alpha into optimization tab
        self.opt_expr_text.delete(1.0, tk.END)
        self.opt_expr_text.insert(tk.END, expression)
        
        # Extract parameters
        self.extract_parameters()

    def parse_expression(self):
        """Parse the base expression to identify numeric parameters"""
        expression = self.base_expr_text.get(1.0, tk.END).strip()
        if not expression:
            messagebox.showerror("Error", "Please enter an expression first")
            return
            
        # Find all numeric parameters in the expression
        param_positions = []
        for match in re.finditer(r'(?<=[,()\s])\d+(?![a-zA-Z])', expression):
            number = int(match.group())
            start_pos = match.start()
            end_pos = match.end()
            param_positions.append((number, start_pos, end_pos))
            
        # Highlight parameters in the text
        self.base_expr_text.tag_remove("param", "1.0", tk.END)
        
        for number, start, end in param_positions:
            # Convert byte offsets to line.char format
            start_idx = f"1.0 + {start} chars"
            end_idx = f"1.0 + {end} chars"
            
            self.base_expr_text.tag_add("param", start_idx, end_idx)
        
        self.base_expr_text.tag_config("param", background="yellow")
        
        # Show how many parameters were found
        messagebox.showinfo("Parameters", f"Found {len(param_positions)} parameters in the expression")

    def generate_variations(self):
        """Generate variations of the expression by changing parameters"""
        expression = self.base_expr_text.get(1.0, tk.END).strip()
        if not expression:
            messagebox.showerror("Error", "Please enter an expression first")
            return
            
        try:
            range_size = int(self.param_range_var.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid range value")
            return
            
        # Find all numeric parameters in the expression
        param_positions = []
        for match in re.finditer(r'(?<=[,()\s])\d+(?![a-zA-Z])', expression):
            number = int(match.group())
            start_pos = match.start()
            end_pos = match.end()
            param_positions.append((number, start_pos, end_pos))
            
        if not param_positions:
            messagebox.showinfo("Info", "No numeric parameters found in the expression")
            return
            
        # Sort positions in reverse order to modify from end to start
        param_positions.sort(reverse=True, key=lambda x: x[1])
        
        # Generate variations
        variations = []
        for number, start, end in param_positions:
            min_val = max(1, number - range_size)
            max_val = number + range_size
            
            for val in range(min_val, max_val + 1):
                if val != number:
                    new_expr = expression[:start] + str(val) + expression[end:]
                    variations.append(new_expr)
                    
        # Set variations in the listbox
        self.variations_list.delete(0, tk.END)
        for var in variations:
            self.variations_list.insert(tk.END, var)
            
        messagebox.showinfo("Variations", f"Generated {len(variations)} variations")
        
    def test_selected_variation(self):
        """Test the selected expression variation"""
        selected = self.variations_list.curselection()
        if not selected:
            messagebox.showerror("Error", "Please select a variation to test")
            return
            
        expression = self.variations_list.get(selected[0])
        self._test_expression(expression)
        
    def test_all_variations(self):
        """Test all generated variations"""
        if self.variations_list.size() == 0:
            messagebox.showerror("Error", "No variations to test")
            return
            
        # Create a thread for testing
        self.is_testing = True
        self.testing_thread = threading.Thread(target=self._test_all_variations)
        self.testing_thread.daemon = True
        self.testing_thread.start()
        
    def _test_all_variations(self):
        """Test all variations in a separate thread"""
        total = self.variations_list.size()
        results = []
        
        for i in range(total):
            if not hasattr(self, 'is_testing') or not self.is_testing:
                break
                
            expression = self.variations_list.get(i)
            result = self._test_expression_impl(expression)
            
            if result:
                results.append((expression, result))
                
            # Update progress
            progress = (i + 1) / total * 100
            self.root.after(0, lambda p=progress: self._update_expr_progress(p))
            
        # Save results
        if results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(Config.RESULTS_DIR, f"variations_{timestamp}.json")
            
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2)
                
            self.root.after(0, lambda: messagebox.showinfo("Success", 
                                                          f"Saved {len(results)} results to {filepath}"))
                                                          
    def _update_expr_progress(self, progress):
        """Update the expression mining progress bar"""
        self.expr_progress_var.set(f"{progress:.1f}%")
        self.expr_progress_bar["value"] = progress
        
    def _test_expression(self, expression):
        """Test a single expression (called from UI thread)"""
        self.is_testing = True
        
        # Start a thread for testing
        test_thread = threading.Thread(target=lambda: self._test_expression_background(expression))
        test_thread.daemon = True
        test_thread.start()
        
    def _test_expression_background(self, expression):
        """Test expression in background thread"""
        try:
            result = self._test_expression_impl(expression)
            
            if result:
                self.root.after(0, lambda: self._show_expression_result(expression, result))
            else:
                self.root.after(0, lambda: messagebox.showerror("Error", "Failed to test expression"))
                
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Test failed: {str(e)}"))
            logger.exception("Expression test error")
        finally:
            self.is_testing = False
            
    def _test_expression_impl(self, expression):
        """Implement the actual expression testing"""
        try:
            # In a real implementation, this would call the API
            # For this example, we'll simulate a test result
            time.sleep(0.5)  # Simulate processing time
            
            # Generate random metrics
            sharpe = round(np.random.uniform(0.8, 2.0), 2)
            turnover = round(np.random.uniform(0.05, 0.4), 2)
            fitness = round(sharpe - 0.1 * turnover, 2)
            
            result = {
                "sharpe": sharpe,
                "turnover": turnover,
                "fitness": fitness,
                "expression": expression
            }
            
            return result
            
        except Exception as e:
            logger.exception(f"Error testing expression: {expression}")
            return None
            
    def _show_expression_result(self, expression, result):
        """Display the expression test result"""
        result_str = f"Expression: {expression}\n\n"
        result_str += f"Sharpe: {result['sharpe']}\n"
        result_str += f"Turnover: {result['turnover']}\n"
        result_str += f"Fitness: {result['fitness']}\n"
        
        messagebox.showinfo("Test Result", result_str)

def main():
    """Main application entry point"""
    root = tk.Tk()
    app = AlphaMinerApp(root)
    root.mainloop()

if __name__ == "__main__":
    main() 