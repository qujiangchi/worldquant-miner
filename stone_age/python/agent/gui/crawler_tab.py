#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Web Crawler tab for the Alpha Agent Network application.
Allows users to crawl websites for financial research.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import json
import os
import threading
import logging
from datetime import datetime
import queue

# Application imports
from agent.config import Config

logger = logging.getLogger(__name__)

class CrawlerFrame(ttk.Frame):
    """Frame for the Web Crawler tab"""
    
    def __init__(self, parent, app):
        """Initialize the frame"""
        super().__init__(parent)
        self.app = app
        self.crawl_queue = queue.Queue()
        self.is_crawling = False
        self.crawl_thread = None
        
        self.setup_ui()
        self.load_saved_targets()
    
    def setup_ui(self):
        """Set up the UI components"""
        # Split into left (control panel) and right (results) frames
        self.paned_window = ttk.PanedWindow(self, orient="horizontal")
        self.paned_window.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Left frame - Control panel
        self.control_frame = ttk.Frame(self.paned_window)
        self.paned_window.add(self.control_frame, weight=1)
        
        # Right frame - Results panel
        self.results_frame = ttk.Frame(self.paned_window)
        self.paned_window.add(self.results_frame, weight=2)
        
        # Set up the control panel
        self.setup_control_panel()
        
        # Set up the results panel
        self.setup_results_panel()
    
    def setup_control_panel(self):
        """Set up the crawler control panel"""
        # Control panel sections
        ttk.Label(self.control_frame, text="Web Crawler", font=("TkDefaultFont", 12, "bold")).pack(anchor="w", padx=5, pady=5)
        
        # URL input section
        url_frame = ttk.LabelFrame(self.control_frame, text="Target URL")
        url_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Label(url_frame, text="URL:").pack(anchor="w", padx=5, pady=5)
        self.url_var = tk.StringVar()
        ttk.Entry(url_frame, textvariable=self.url_var, width=40).pack(fill="x", padx=5, pady=5)
        
        # URL buttons
        url_button_frame = ttk.Frame(url_frame)
        url_button_frame.pack(fill="x", padx=5, pady=5)
        ttk.Button(url_button_frame, text="Add to Queue", command=self.add_to_queue).pack(side="left", padx=5)
        ttk.Button(url_button_frame, text="Start Crawling", command=self.start_crawling).pack(side="left", padx=5)
        ttk.Button(url_button_frame, text="Stop Crawling", command=self.stop_crawling).pack(side="left", padx=5)
        
        # Predefined targets section
        targets_frame = ttk.LabelFrame(self.control_frame, text="Saved Targets")
        targets_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Targets listbox with scrollbar
        targets_list_frame = ttk.Frame(targets_frame)
        targets_list_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.targets_listbox = tk.Listbox(targets_list_frame, height=10)
        self.targets_listbox.pack(side="left", fill="both", expand=True)
        
        targets_scrollbar = ttk.Scrollbar(targets_list_frame, orient="vertical", command=self.targets_listbox.yview)
        self.targets_listbox.configure(yscrollcommand=targets_scrollbar.set)
        targets_scrollbar.pack(side="right", fill="y")
        
        # Bind double-click to select target
        self.targets_listbox.bind("<Double-1>", self.select_target)
        
        # Target buttons
        target_button_frame = ttk.Frame(targets_frame)
        target_button_frame.pack(fill="x", padx=5, pady=5)
        ttk.Button(target_button_frame, text="Add Current", command=self.save_target).pack(side="left", padx=5)
        ttk.Button(target_button_frame, text="Remove Selected", command=self.remove_target).pack(side="left", padx=5)
        ttk.Button(target_button_frame, text="Load Defaults", command=self.load_default_targets).pack(side="left", padx=5)
        
        # Crawler configuration
        config_frame = ttk.LabelFrame(self.control_frame, text="Crawler Configuration")
        config_frame.pack(fill="x", padx=5, pady=5)
        
        # Delay between requests
        delay_frame = ttk.Frame(config_frame)
        delay_frame.pack(fill="x", padx=5, pady=5)
        ttk.Label(delay_frame, text="Delay (seconds):").pack(side="left")
        self.delay_var = tk.DoubleVar(value=Config.CRAWL_DELAY)
        ttk.Spinbox(delay_frame, from_=0.1, to=10.0, increment=0.1, textvariable=self.delay_var, width=5).pack(side="left", padx=5)
        
        # Max pages per site
        max_pages_frame = ttk.Frame(config_frame)
        max_pages_frame.pack(fill="x", padx=5, pady=5)
        ttk.Label(max_pages_frame, text="Max Pages:").pack(side="left")
        self.max_pages_var = tk.IntVar(value=Config.MAX_PAGES_PER_SITE)
        ttk.Spinbox(max_pages_frame, from_=1, to=1000, textvariable=self.max_pages_var, width=5).pack(side="left", padx=5)
        
        # User agent
        agent_frame = ttk.Frame(config_frame)
        agent_frame.pack(fill="x", padx=5, pady=5)
        ttk.Label(agent_frame, text="User Agent:").pack(side="left")
        self.user_agent_var = tk.StringVar(value=Config.USER_AGENT)
        ttk.Entry(agent_frame, textvariable=self.user_agent_var).pack(side="left", fill="x", expand=True, padx=5)
    
    def setup_results_panel(self):
        """Set up the results panel"""
        # Results panel sections
        results_frame = ttk.Frame(self.results_frame)
        results_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Status section
        status_frame = ttk.Frame(results_frame)
        status_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Label(status_frame, text="Status:").pack(side="left")
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(status_frame, textvariable=self.status_var).pack(side="left", padx=5)
        
        # Progress bar
        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_bar = ttk.Progressbar(results_frame, variable=self.progress_var, mode="determinate")
        self.progress_bar.pack(fill="x", padx=5, pady=5)
        
        # Stats section
        stats_frame = ttk.Frame(results_frame)
        stats_frame.pack(fill="x", padx=5, pady=5)
        
        self.pages_crawled_var = tk.StringVar(value="Pages Crawled: 0")
        ttk.Label(stats_frame, textvariable=self.pages_crawled_var).pack(side="left", padx=5)
        
        self.queue_size_var = tk.StringVar(value="Queue Size: 0")
        ttk.Label(stats_frame, textvariable=self.queue_size_var).pack(side="left", padx=5)
        
        self.time_elapsed_var = tk.StringVar(value="Time: 00:00:00")
        ttk.Label(stats_frame, textvariable=self.time_elapsed_var).pack(side="left", padx=5)
        
        # Results notebook with tabs for different views
        self.results_notebook = ttk.Notebook(results_frame)
        self.results_notebook.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Log tab
        self.log_frame = ttk.Frame(self.results_notebook)
        self.results_notebook.add(self.log_frame, text="Crawler Log")
        
        self.log_text = scrolledtext.ScrolledText(self.log_frame, wrap=tk.WORD)
        self.log_text.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Pages tab
        self.pages_frame = ttk.Frame(self.results_notebook)
        self.results_notebook.add(self.pages_frame, text="Crawled Pages")
        
        # Pages tree with columns for URL, title, and timestamp
        self.pages_tree = ttk.Treeview(self.pages_frame, columns=("url", "title", "timestamp"), show="headings")
        self.pages_tree.heading("url", text="URL")
        self.pages_tree.heading("title", text="Title")
        self.pages_tree.heading("timestamp", text="Timestamp")
        
        self.pages_tree.column("url", width=300)
        self.pages_tree.column("title", width=200)
        self.pages_tree.column("timestamp", width=100)
        
        pages_scrollbar = ttk.Scrollbar(self.pages_frame, orient="vertical", command=self.pages_tree.yview)
        self.pages_tree.configure(yscrollcommand=pages_scrollbar.set)
        
        self.pages_tree.pack(side="left", fill="both", expand=True)
        pages_scrollbar.pack(side="right", fill="y")
        
        # Extracted tab for content analysis
        self.extracted_frame = ttk.Frame(self.results_notebook)
        self.results_notebook.add(self.extracted_frame, text="Extracted Content")
        
        self.extracted_text = scrolledtext.ScrolledText(self.extracted_frame, wrap=tk.WORD)
        self.extracted_text.pack(fill="both", expand=True, padx=5, pady=5)
    
    def load_saved_targets(self):
        """Load saved crawl targets from file"""
        targets_file = os.path.join(Config.DATA_DIR, "crawl_targets.json")
        
        if os.path.exists(targets_file):
            try:
                with open(targets_file, 'r') as f:
                    targets = json.load(f)
                
                for target in targets:
                    self.targets_listbox.insert(tk.END, target)
            except Exception as e:
                logger.error(f"Error loading saved targets: {e}")
    
    def save_targets(self):
        """Save current targets to file"""
        targets_file = os.path.join(Config.DATA_DIR, "crawl_targets.json")
        
        try:
            targets = list(self.targets_listbox.get(0, tk.END))
            
            os.makedirs(os.path.dirname(targets_file), exist_ok=True)
            with open(targets_file, 'w') as f:
                json.dump(targets, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving targets: {e}")
    
    def load_default_targets(self):
        """Load default research sources as targets"""
        # Clear current targets
        self.targets_listbox.delete(0, tk.END)
        
        # Add default sources
        for source in Config.DEFAULT_SOURCES:
            self.targets_listbox.insert(tk.END, source)
        
        # Save the updated targets
        self.save_targets()
        
        # Show success message
        self.app.set_status("Default research sources loaded")
    
    def select_target(self, event):
        """Select a target from the listbox"""
        selection = self.targets_listbox.curselection()
        if selection:
            target = self.targets_listbox.get(selection[0])
            self.url_var.set(target)
    
    def save_target(self):
        """Save current URL as a target"""
        url = self.url_var.get().strip()
        
        if not url:
            messagebox.showwarning("Empty URL", "Please enter a URL to save")
            return
        
        # Check if already exists
        existing_targets = list(self.targets_listbox.get(0, tk.END))
        if url in existing_targets:
            messagebox.showinfo("Duplicate", f"URL '{url}' is already in the targets list")
            return
        
        # Add to listbox
        self.targets_listbox.insert(tk.END, url)
        
        # Save targets
        self.save_targets()
        
        # Show success message
        self.app.set_status(f"Added '{url}' to crawl targets")
    
    def remove_target(self):
        """Remove selected target from the listbox"""
        selection = self.targets_listbox.curselection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a target to remove")
            return
        
        # Get target before removing
        target = self.targets_listbox.get(selection[0])
        
        # Remove from listbox
        self.targets_listbox.delete(selection[0])
        
        # Save targets
        self.save_targets()
        
        # Show success message
        self.app.set_status(f"Removed '{target}' from crawl targets")
    
    def add_to_queue(self):
        """Add current URL to the crawl queue"""
        url = self.url_var.get().strip()
        
        if not url:
            messagebox.showwarning("Empty URL", "Please enter a URL to add to the queue")
            return
        
        # Add to queue
        self.crawl_queue.put(url)
        
        # Update queue size
        self.queue_size_var.set(f"Queue Size: {self.crawl_queue.qsize()}")
        
        # Log the action
        self.log_message(f"Added to queue: {url}")
        
        # Show success message
        self.app.set_status(f"Added '{url}' to crawl queue")
    
    def start_crawling(self):
        """Start the crawling process"""
        if self.is_crawling:
            messagebox.showinfo("Already Crawling", "Crawler is already running")
            return
        
        if self.crawl_queue.empty():
            # If queue is empty, add current URL if available
            url = self.url_var.get().strip()
            if url:
                self.crawl_queue.put(url)
            else:
                messagebox.showwarning("Empty Queue", "Please add at least one URL to the crawl queue")
                return
        
        # Update UI
        self.is_crawling = True
        self.status_var.set("Crawling...")
        self.progress_var.set(0)
        
        # Clear previous results
        self.log_text.delete("1.0", tk.END)
        for item in self.pages_tree.get_children():
            self.pages_tree.delete(item)
        
        # Reset stats
        self.pages_crawled_var.set("Pages Crawled: 0")
        self.queue_size_var.set(f"Queue Size: {self.crawl_queue.qsize()}")
        self.time_elapsed_var.set("Time: 00:00:00")
        
        # Log start
        self.log_message("Crawler started")
        
        # Start crawling in a background thread
        self.crawl_thread = threading.Thread(target=self.crawl_worker)
        self.crawl_thread.daemon = True
        self.crawl_thread.start()
        
        # Start timer update
        self.start_time = datetime.now()
        self.update_timer()
    
    def stop_crawling(self):
        """Stop the crawling process"""
        if not self.is_crawling:
            messagebox.showinfo("Not Crawling", "Crawler is not running")
            return
        
        # Update flag to stop thread
        self.is_crawling = False
        
        # Log stop
        self.log_message("Crawler stopping... (waiting for current page to complete)")
        
        # Update status
        self.status_var.set("Stopping...")
    
    def crawl_worker(self):
        """Worker thread for crawling websites"""
        try:
            # Import crawler functionality here to avoid circular imports
            # This is a placeholder for actual crawler implementation
            pages_crawled = 0
            max_pages = self.max_pages_var.get()
            delay = self.delay_var.get()
            
            # Mock crawling process
            while self.is_crawling and not self.crawl_queue.empty() and pages_crawled < max_pages:
                url = self.crawl_queue.get()
                
                # Log current action
                self.log_message(f"Crawling: {url}")
                
                # Simulate crawling
                import time
                import random
                time.sleep(delay)  # Respect crawl delay
                
                # Simulate finding a title
                title = f"Page Title {pages_crawled + 1}"
                
                # Add to results
                self.add_crawled_page(url, title)
                
                # Simulate finding links and adding to queue
                if pages_crawled < max_pages - 1:
                    num_links = random.randint(1, 3)
                    for i in range(num_links):
                        new_url = f"{url}/subpage_{i}"
                        self.crawl_queue.put(new_url)
                        self.log_message(f"Found link: {new_url}")
                    
                    # Update queue size
                    self.update_queue_size()
                
                # Update progress
                pages_crawled += 1
                self.update_progress(pages_crawled, max_pages)
                
            # Finished crawling
            if not self.is_crawling:
                self.log_message("Crawler stopped by user")
            elif self.crawl_queue.empty():
                self.log_message("Crawler finished: Queue empty")
            else:
                self.log_message(f"Crawler finished: Reached max pages ({max_pages})")
        
        except Exception as e:
            self.log_message(f"Crawler error: {e}")
            logger.exception("Error in crawler worker thread")
        
        finally:
            # Update UI
            self.is_crawling = False
            self.status_var.set("Ready")
            
            # Show finished message
            self.app.set_status(f"Crawling finished: {pages_crawled} pages crawled")
    
    def log_message(self, message):
        """Log a message to the crawler log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        
        # Schedule the UI update to run on the main thread
        self.after(0, lambda: self.log_text.insert(tk.END, log_entry))
        self.after(0, lambda: self.log_text.see(tk.END))
    
    def add_crawled_page(self, url, title):
        """Add a crawled page to the results"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Schedule the UI update to run on the main thread
        self.after(0, lambda: self.pages_tree.insert("", "end", values=(url, title, timestamp)))
        
        # Update pages crawled counter
        pages_crawled = len(self.pages_tree.get_children())
        self.after(0, lambda: self.pages_crawled_var.set(f"Pages Crawled: {pages_crawled}"))
    
    def update_progress(self, current, total):
        """Update the progress bar"""
        progress = (current / total) * 100 if total > 0 else 0
        self.after(0, lambda: self.progress_var.set(progress))
    
    def update_queue_size(self):
        """Update the queue size display"""
        self.after(0, lambda: self.queue_size_var.set(f"Queue Size: {self.crawl_queue.qsize()}"))
    
    def update_timer(self):
        """Update the elapsed time display"""
        if self.is_crawling:
            elapsed = datetime.now() - self.start_time
            hours, remainder = divmod(elapsed.total_seconds(), 3600)
            minutes, seconds = divmod(remainder, 60)
            time_str = f"Time: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
            self.time_elapsed_var.set(time_str)
            
            # Schedule next update in 1 second
            self.after(1000, self.update_timer) 