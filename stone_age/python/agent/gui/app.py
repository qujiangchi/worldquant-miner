#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main GUI application for the Alpha Agent Network.
Defines the main window, tabs, and controls for the application.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import threading
import logging
import os
import json
import time
from datetime import datetime

# Application imports
from agent.config import Config
from agent.gui.agent_designer import AgentDesignerFrame
from agent.gui.crawler_tab import CrawlerFrame
from agent.gui.alpha_generator_tab import AlphaGeneratorFrame
from agent.gui.research_tab import ResearchFrame
from agent.gui.settings_tab import SettingsFrame

# Configure logging
logger = logging.getLogger(__name__)

class AlphaAgentApp:
    """Main application class for the Alpha Agent Network"""
    
    def __init__(self):
        """Initialize the application"""
        self.root = tk.Tk()
        self.root.title(Config.WINDOW_TITLE)
        self.root.geometry(Config.WINDOW_SIZE)
        self.root.minsize(800, 600)
        
        # Set application icon
        try:
            icon_path = os.path.join(os.path.dirname(__file__), "assets", "icon.png")
            if os.path.exists(icon_path):
                icon = tk.PhotoImage(file=icon_path)
                self.root.iconphoto(True, icon)
        except Exception as e:
            logger.error(f"Failed to set application icon: {e}")
        
        # Initialize the UI
        self.setup_ui()
        
        # Initialize variables
        self.is_running = False
        self.active_agents = {}
        
        # Load user settings
        self.load_settings()
    
    def setup_ui(self):
        """Set up the main UI components"""
        # Create main notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Create the tabs
        self.create_agent_designer_tab()
        self.create_crawler_tab()
        self.create_research_tab()
        self.create_alpha_generator_tab()
        self.create_settings_tab()
        
        # Create status bar
        self.create_status_bar()
        
        # Bind events
        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_changed)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
    
    def create_agent_designer_tab(self):
        """Create the Agent Designer tab"""
        self.agent_designer_frame = AgentDesignerFrame(self.notebook, self)
        self.notebook.add(self.agent_designer_frame, text="Agent Designer")
    
    def create_crawler_tab(self):
        """Create the Web Crawler tab"""
        self.crawler_frame = CrawlerFrame(self.notebook, self)
        self.notebook.add(self.crawler_frame, text="Web Crawler")
    
    def create_research_tab(self):
        """Create the Research tab"""
        self.research_frame = ResearchFrame(self.notebook, self)
        self.notebook.add(self.research_frame, text="Research")
    
    def create_alpha_generator_tab(self):
        """Create the Alpha Generator tab"""
        self.alpha_generator_frame = AlphaGeneratorFrame(self.notebook, self)
        self.notebook.add(self.alpha_generator_frame, text="Alpha Generator")
    
    def create_settings_tab(self):
        """Create the Settings tab"""
        self.settings_frame = SettingsFrame(self.notebook, self)
        self.notebook.add(self.settings_frame, text="Settings")
    
    def create_status_bar(self):
        """Create the status bar at the bottom of the window"""
        status_frame = ttk.Frame(self.root)
        status_frame.pack(side="bottom", fill="x")
        
        # Add a separator above the status bar
        ttk.Separator(self.root, orient="horizontal").pack(side="bottom", fill="x")
        
        # Status text
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(status_frame, textvariable=self.status_var, anchor="w")
        status_label.pack(side="left", padx=5)
        
        # Active agent counter
        self.agent_count_var = tk.StringVar(value="Active Agents: 0")
        agent_count_label = ttk.Label(status_frame, textvariable=self.agent_count_var)
        agent_count_label.pack(side="right", padx=5)
    
    def set_status(self, message):
        """Set the status bar message"""
        self.status_var.set(message)
        self.root.update_idletasks()
    
    def update_agent_count(self):
        """Update the active agent count in the status bar"""
        count = len([a for a in self.active_agents.values() if a.get('active', False)])
        self.agent_count_var.set(f"Active Agents: {count}")
    
    def on_tab_changed(self, event):
        """Handle tab change events"""
        current_tab = self.notebook.index(self.notebook.select())
        tab_name = self.notebook.tab(current_tab, "text")
        self.set_status(f"Current tab: {tab_name}")
    
    def on_close(self):
        """Clean up resources and exit the application"""
        # Stop any running agents
        for agent_id, agent_info in self.active_agents.items():
            if agent_info.get('active', False):
                try:
                    agent_info['instance'].stop()
                except Exception as e:
                    logger.error(f"Error stopping agent {agent_id}: {e}")
        
        # Save settings
        self.save_settings()
        
        # Destroy the window
        self.root.destroy()
    
    def load_settings(self):
        """Load user settings"""
        # Config is already loaded from the Config class
        pass
    
    def save_settings(self):
        """Save user settings"""
        # Config is already saved in the settings frame
        if hasattr(self, 'settings_frame'):
            self.settings_frame.save_settings()
    
    def start(self):
        """Start the application main loop"""
        self.root.mainloop()

if __name__ == "__main__":
    app = AlphaAgentApp()
    app.start() 