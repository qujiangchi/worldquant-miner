#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Settings tab for the Alpha Agent Network application.
This tab allows users to configure application settings.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import logging
import os
import json

# Application imports
from agent.config import Config

# Configure logging
logger = logging.getLogger(__name__)

class SettingsFrame(ttk.Frame):
    """Settings tab frame for the Alpha Agent Network application"""
    
    def __init__(self, parent, controller):
        """Initialize the Settings frame"""
        super().__init__(parent)
        self.controller = controller
        
        # Set up the UI
        self.setup_ui()
        
        # Load current settings
        self.load_settings()
    
    def setup_ui(self):
        """Set up the settings tab UI"""
        # Use grid layout with weights
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)  # Content row gets all extra space
        
        # Title label
        title_label = ttk.Label(
            self, 
            text="Application Settings",
            font=("Arial", 14, "bold")
        )
        title_label.grid(row=0, column=0, padx=10, pady=(10, 5), sticky="w")
        
        # Create notebook for settings categories
        self.settings_notebook = ttk.Notebook(self)
        self.settings_notebook.grid(row=1, column=0, padx=10, pady=5, sticky="nsew")
        
        # Create settings category tabs
        self.create_general_settings()
        self.create_agent_settings()
        self.create_research_settings()
        self.create_alpha_settings()
        
        # Control buttons at the bottom
        self.create_control_buttons()
    
    def create_general_settings(self):
        """Create general application settings tab"""
        general_frame = ttk.Frame(self.settings_notebook)
        self.settings_notebook.add(general_frame, text="General")
        
        # Use grid layout with padding
        for i in range(4):
            general_frame.columnconfigure(i, weight=1)
        
        # App theme setting
        ttk.Label(general_frame, text="Application Theme:").grid(
            row=0, column=0, padx=5, pady=5, sticky="w"
        )
        self.theme_var = tk.StringVar()
        theme_combo = ttk.Combobox(
            general_frame, 
            textvariable=self.theme_var,
            values=["Light", "Dark", "System"]
        )
        theme_combo.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        
        # Data directory setting
        ttk.Label(general_frame, text="Data Directory:").grid(
            row=1, column=0, padx=5, pady=5, sticky="w"
        )
        self.data_dir_var = tk.StringVar()
        data_dir_entry = ttk.Entry(
            general_frame, 
            textvariable=self.data_dir_var
        )
        data_dir_entry.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        
        browse_button = ttk.Button(
            general_frame,
            text="Browse...",
            command=self.browse_data_dir
        )
        browse_button.grid(row=1, column=2, padx=5, pady=5)
        
        # Log level setting
        ttk.Label(general_frame, text="Log Level:").grid(
            row=2, column=0, padx=5, pady=5, sticky="w"
        )
        self.log_level_var = tk.StringVar()
        log_level_combo = ttk.Combobox(
            general_frame, 
            textvariable=self.log_level_var,
            values=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        )
        log_level_combo.grid(row=2, column=1, padx=5, pady=5, sticky="ew")
        
        # Auto-save setting
        ttk.Label(general_frame, text="Auto-save Interval (minutes):").grid(
            row=3, column=0, padx=5, pady=5, sticky="w"
        )
        self.autosave_var = tk.StringVar()
        autosave_entry = ttk.Entry(
            general_frame, 
            textvariable=self.autosave_var,
            width=5
        )
        autosave_entry.grid(row=3, column=1, padx=5, pady=5, sticky="w")
    
    def create_agent_settings(self):
        """Create agent-specific settings tab"""
        agent_frame = ttk.Frame(self.settings_notebook)
        self.settings_notebook.add(agent_frame, text="Agents")
        
        # Use grid layout with padding
        for i in range(2):
            agent_frame.columnconfigure(i, weight=1)
        
        # Max agents setting
        ttk.Label(agent_frame, text="Maximum Concurrent Agents:").grid(
            row=0, column=0, padx=5, pady=5, sticky="w"
        )
        self.max_agents_var = tk.StringVar()
        max_agents_entry = ttk.Entry(
            agent_frame, 
            textvariable=self.max_agents_var,
            width=5
        )
        max_agents_entry.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        
        # Agent memory limit
        ttk.Label(agent_frame, text="Agent Memory Limit (MB):").grid(
            row=1, column=0, padx=5, pady=5, sticky="w"
        )
        self.agent_memory_var = tk.StringVar()
        agent_memory_entry = ttk.Entry(
            agent_frame, 
            textvariable=self.agent_memory_var,
            width=5
        )
        agent_memory_entry.grid(row=1, column=1, padx=5, pady=5, sticky="w")
        
        # Default agent model
        ttk.Label(agent_frame, text="Default Agent Model:").grid(
            row=2, column=0, padx=5, pady=5, sticky="w"
        )
        self.agent_model_var = tk.StringVar()
        agent_model_combo = ttk.Combobox(
            agent_frame, 
            textvariable=self.agent_model_var,
            values=["GPT-3.5", "GPT-4", "Claude", "Custom"]
        )
        agent_model_combo.grid(row=2, column=1, padx=5, pady=5, sticky="ew")
        
        # Enable agent logging
        self.agent_logging_var = tk.BooleanVar()
        agent_logging_check = ttk.Checkbutton(
            agent_frame,
            text="Enable Agent Logging",
            variable=self.agent_logging_var
        )
        agent_logging_check.grid(row=3, column=0, columnspan=2, padx=5, pady=5, sticky="w")
        
        # Auto-restart failed agents
        self.agent_restart_var = tk.BooleanVar()
        agent_restart_check = ttk.Checkbutton(
            agent_frame,
            text="Auto-restart Failed Agents",
            variable=self.agent_restart_var
        )
        agent_restart_check.grid(row=4, column=0, columnspan=2, padx=5, pady=5, sticky="w")
    
    def create_research_settings(self):
        """Create research-specific settings tab"""
        research_frame = ttk.Frame(self.settings_notebook)
        self.settings_notebook.add(research_frame, text="Research")
        
        # Simple placeholder for now
        placeholder = ttk.Label(
            research_frame, 
            text="Research Settings (Under Development)",
            font=("Arial", 12)
        )
        placeholder.pack(pady=50)
    
    def create_alpha_settings(self):
        """Create alpha generator settings tab"""
        alpha_frame = ttk.Frame(self.settings_notebook)
        self.settings_notebook.add(alpha_frame, text="Alpha Generator")
        
        # Simple placeholder for now
        placeholder = ttk.Label(
            alpha_frame, 
            text="Alpha Generator Settings (Under Development)",
            font=("Arial", 12)
        )
        placeholder.pack(pady=50)
    
    def create_control_buttons(self):
        """Create control buttons at the bottom of the settings tab"""
        button_frame = ttk.Frame(self)
        button_frame.grid(row=2, column=0, padx=10, pady=10, sticky="ew")
        
        # Save button
        save_button = ttk.Button(
            button_frame,
            text="Save Settings",
            command=self.save_settings
        )
        save_button.pack(side="right", padx=5)
        
        # Reset to defaults button
        reset_button = ttk.Button(
            button_frame,
            text="Reset to Defaults",
            command=self.reset_settings
        )
        reset_button.pack(side="right", padx=5)
    
    def browse_data_dir(self):
        """Open directory browser for data directory selection"""
        directory = filedialog.askdirectory(
            initialdir=self.data_dir_var.get()
        )
        if directory:
            self.data_dir_var.set(directory)
    
    def load_settings(self):
        """Load current settings into the UI"""
        # Use Config class values as defaults
        self.theme_var.set(getattr(Config, "THEME", "Light"))
        self.data_dir_var.set(getattr(Config, "DATA_DIR", os.path.join(os.path.dirname(__file__), "..", "data")))
        self.log_level_var.set(getattr(Config, "LOG_LEVEL", "INFO"))
        self.autosave_var.set(str(getattr(Config, "AUTOSAVE_INTERVAL", "5")))
        
        self.max_agents_var.set(str(getattr(Config, "MAX_AGENTS", "5")))
        self.agent_memory_var.set(str(getattr(Config, "AGENT_MEMORY_LIMIT", "1024")))
        self.agent_model_var.set(getattr(Config, "DEFAULT_AGENT_MODEL", "GPT-4"))
        self.agent_logging_var.set(getattr(Config, "ENABLE_AGENT_LOGGING", True))
        self.agent_restart_var.set(getattr(Config, "AUTO_RESTART_AGENTS", False))
    
    def save_settings(self):
        """Save settings to config file"""
        try:
            # Update Config class attributes
            Config.THEME = self.theme_var.get()
            Config.DATA_DIR = self.data_dir_var.get()
            Config.LOG_LEVEL = self.log_level_var.get()
            Config.AUTOSAVE_INTERVAL = int(self.autosave_var.get())
            
            Config.MAX_AGENTS = int(self.max_agents_var.get())
            Config.AGENT_MEMORY_LIMIT = int(self.agent_memory_var.get())
            Config.DEFAULT_AGENT_MODEL = self.agent_model_var.get()
            Config.ENABLE_AGENT_LOGGING = self.agent_logging_var.get()
            Config.AUTO_RESTART_AGENTS = self.agent_restart_var.get()
            
            # Save to config file
            Config.save_config()
            
            # Show success message
            self.controller.set_status("Settings saved successfully")
            logger.info("Settings saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving settings: {e}")
            messagebox.showerror(
                "Error", 
                f"Failed to save settings: {e}"
            )
    
    def reset_settings(self):
        """Reset settings to default values"""
        if messagebox.askyesno(
            "Confirm Reset", 
            "Are you sure you want to reset all settings to default values?"
        ):
            Config.reset_to_defaults()
            self.load_settings()
            self.controller.set_status("Settings reset to defaults")
            logger.info("Settings reset to defaults") 