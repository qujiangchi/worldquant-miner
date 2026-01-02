#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Alpha Generator tab for the Alpha Agent Network application.
Allows users to generate trading alphas based on research.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import json
import os
import logging
import threading
from datetime import datetime

# Application imports
from agent.config import Config

logger = logging.getLogger(__name__)

class AlphaGeneratorFrame(ttk.Frame):
    """Frame for the Alpha Generator tab"""
    
    def __init__(self, parent, app):
        """Initialize the frame"""
        super().__init__(parent)
        self.app = app
        self.alpha_being_generated = False
        
        self.setup_ui()
        self.load_saved_alphas()
    
    def setup_ui(self):
        """Set up the UI components"""
        # Create a paned window with left and right sides
        self.paned_window = ttk.PanedWindow(self, orient="horizontal")
        self.paned_window.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Left frame for alpha list and controls
        self.left_frame = ttk.Frame(self.paned_window)
        self.paned_window.add(self.left_frame, weight=1)
        
        # Right frame for alpha details and editing
        self.right_frame = ttk.Frame(self.paned_window)
        self.paned_window.add(self.right_frame, weight=2)
        
        # Set up the left frame
        self.setup_left_frame()
        
        # Set up the right frame
        self.setup_right_frame()
    
    def setup_left_frame(self):
        """Set up the left frame with alpha list and controls"""
        # Title
        ttk.Label(self.left_frame, text="Alpha Generator", font=("TkDefaultFont", 12, "bold")).pack(anchor="w", padx=5, pady=5)
        
        # Alpha list frame
        alpha_list_frame = ttk.LabelFrame(self.left_frame, text="Generated Alphas")
        alpha_list_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Alpha list with scrollbar
        list_container = ttk.Frame(alpha_list_frame)
        list_container.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.alpha_listbox = tk.Listbox(list_container, height=15)
        self.alpha_listbox.pack(side="left", fill="both", expand=True)
        
        scrollbar = ttk.Scrollbar(list_container, orient="vertical", command=self.alpha_listbox.yview)
        self.alpha_listbox.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        
        # Bind selection event
        self.alpha_listbox.bind("<<ListboxSelect>>", self.on_alpha_selected)
        
        # Control buttons
        control_frame = ttk.Frame(self.left_frame)
        control_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Button(control_frame, text="Generate New Alpha", command=self.generate_alpha).pack(side="left", padx=5)
        ttk.Button(control_frame, text="Delete Alpha", command=self.delete_alpha).pack(side="left", padx=5)
        ttk.Button(control_frame, text="Backtest Alpha", command=self.backtest_alpha).pack(side="left", padx=5)
    
    def setup_right_frame(self):
        """Set up the right frame with alpha details and editor"""
        # Create notebook for multiple tabs
        self.alpha_notebook = ttk.Notebook(self.right_frame)
        self.alpha_notebook.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Alpha details tab
        self.details_frame = ttk.Frame(self.alpha_notebook)
        self.alpha_notebook.add(self.details_frame, text="Alpha Details")
        
        self.setup_details_tab()
        
        # Alpha editor tab
        self.editor_frame = ttk.Frame(self.alpha_notebook)
        self.alpha_notebook.add(self.editor_frame, text="Alpha Editor")
        
        self.setup_editor_tab()
        
        # Backtest results tab
        self.backtest_frame = ttk.Frame(self.alpha_notebook)
        self.alpha_notebook.add(self.backtest_frame, text="Backtest Results")
        
        self.setup_backtest_tab()
    
    def setup_details_tab(self):
        """Set up the alpha details tab"""
        # Create a form for alpha details
        form_frame = ttk.Frame(self.details_frame)
        form_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Alpha name
        ttk.Label(form_frame, text="Alpha Name:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.name_var = tk.StringVar()
        ttk.Entry(form_frame, textvariable=self.name_var).grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        
        # Alpha description
        ttk.Label(form_frame, text="Description:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.description_text = scrolledtext.ScrolledText(form_frame, height=5, width=40)
        self.description_text.grid(row=1, column=1, sticky="ew", padx=5, pady=5)
        
        # Alpha expression
        ttk.Label(form_frame, text="Expression:").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        self.expression_var = tk.StringVar()
        ttk.Entry(form_frame, textvariable=self.expression_var, width=50).grid(row=2, column=1, sticky="ew", padx=5, pady=5)
        
        # Alpha metadata
        ttk.Label(form_frame, text="Created:").grid(row=3, column=0, sticky="w", padx=5, pady=5)
        self.created_var = tk.StringVar()
        ttk.Label(form_frame, textvariable=self.created_var).grid(row=3, column=1, sticky="w", padx=5, pady=5)
        
        ttk.Label(form_frame, text="Last Modified:").grid(row=4, column=0, sticky="w", padx=5, pady=5)
        self.modified_var = tk.StringVar()
        ttk.Label(form_frame, textvariable=self.modified_var).grid(row=4, column=1, sticky="w", padx=5, pady=5)
        
        # Alpha tags
        ttk.Label(form_frame, text="Tags:").grid(row=5, column=0, sticky="w", padx=5, pady=5)
        self.tags_var = tk.StringVar()
        ttk.Entry(form_frame, textvariable=self.tags_var).grid(row=5, column=1, sticky="ew", padx=5, pady=5)
        
        # Alpha status
        ttk.Label(form_frame, text="Status:").grid(row=6, column=0, sticky="w", padx=5, pady=5)
        self.status_var = tk.StringVar()
        ttk.Label(form_frame, textvariable=self.status_var).grid(row=6, column=1, sticky="w", padx=5, pady=5)
        
        # Make columns expandable
        form_frame.columnconfigure(1, weight=1)
        
        # Save button
        save_frame = ttk.Frame(self.details_frame)
        save_frame.pack(fill="x", padx=10, pady=10)
        ttk.Button(save_frame, text="Save Changes", command=self.save_alpha).pack(side="right")
    
    def setup_editor_tab(self):
        """Set up the alpha editor tab"""
        # Instructions
        ttk.Label(self.editor_frame, text="Edit Alpha Expression:").pack(anchor="w", padx=10, pady=5)
        
        # Expression editor
        self.editor_text = scrolledtext.ScrolledText(self.editor_frame, height=15, width=80)
        self.editor_text.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Button frame
        button_frame = ttk.Frame(self.editor_frame)
        button_frame.pack(fill="x", padx=10, pady=10)
        
        ttk.Button(button_frame, text="Validate", command=self.validate_expression).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Apply Changes", command=self.apply_editor_changes).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Reset", command=self.reset_editor).pack(side="left", padx=5)
    
    def setup_backtest_tab(self):
        """Set up the backtest results tab"""
        # Backtest settings frame
        settings_frame = ttk.LabelFrame(self.backtest_frame, text="Backtest Settings")
        settings_frame.pack(fill="x", padx=10, pady=10)
        
        # Start date
        ttk.Label(settings_frame, text="Start Date:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.start_date_var = tk.StringVar(value="2018-01-01")
        ttk.Entry(settings_frame, textvariable=self.start_date_var, width=12).grid(row=0, column=1, sticky="w", padx=5, pady=5)
        
        # End date
        ttk.Label(settings_frame, text="End Date:").grid(row=0, column=2, sticky="w", padx=5, pady=5)
        self.end_date_var = tk.StringVar(value="2022-12-31")
        ttk.Entry(settings_frame, textvariable=self.end_date_var, width=12).grid(row=0, column=3, sticky="w", padx=5, pady=5)
        
        # Universe
        ttk.Label(settings_frame, text="Universe:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.universe_var = tk.StringVar(value="US Equities")
        universes = ["US Equities", "Global Equities", "Crypto", "Forex", "Futures"]
        ttk.Combobox(settings_frame, textvariable=self.universe_var, values=universes).grid(row=1, column=1, columnspan=3, sticky="ew", padx=5, pady=5)
        
        # Run backtest button
        ttk.Button(settings_frame, text="Run Backtest", command=self.run_backtest).grid(row=2, column=0, columnspan=4, padx=5, pady=5)
        
        # Results notebook
        results_notebook = ttk.Notebook(self.backtest_frame)
        results_notebook.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Performance tab
        performance_frame = ttk.Frame(results_notebook)
        results_notebook.add(performance_frame, text="Performance")
        
        # Placeholder for performance chart
        ttk.Label(performance_frame, text="Performance chart will appear here").pack(padx=10, pady=10)
        
        # Statistics tab
        stats_frame = ttk.Frame(results_notebook)
        results_notebook.add(stats_frame, text="Statistics")
        
        # Stats text
        self.stats_text = scrolledtext.ScrolledText(stats_frame, height=10, width=80)
        self.stats_text.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Set some default stats text
        default_stats = "Run a backtest to see statistics here."
        self.stats_text.insert(tk.END, default_stats)
        self.stats_text.config(state="disabled")
    
    def load_saved_alphas(self):
        """Load saved alphas from disk"""
        alpha_dir = Config.ALPHA_DIR
        os.makedirs(alpha_dir, exist_ok=True)
        
        # Clear listbox
        self.alpha_listbox.delete(0, tk.END)
        
        # Load alpha files
        try:
            for filename in os.listdir(alpha_dir):
                if filename.endswith('.json'):
                    alpha_name = filename[:-5]  # Remove .json extension
                    self.alpha_listbox.insert(tk.END, alpha_name)
        except Exception as e:
            logger.error(f"Error loading alphas: {e}")
    
    def on_alpha_selected(self, event):
        """Handler for alpha selection in the listbox"""
        selection = self.alpha_listbox.curselection()
        if not selection:
            return
        
        alpha_name = self.alpha_listbox.get(selection[0])
        self.load_alpha_details(alpha_name)
    
    def load_alpha_details(self, alpha_name):
        """Load details of the selected alpha"""
        alpha_file = os.path.join(Config.ALPHA_DIR, f"{alpha_name}.json")
        
        try:
            with open(alpha_file, 'r') as f:
                alpha_data = json.load(f)
            
            # Update details tab
            self.name_var.set(alpha_data.get('name', ''))
            self.description_text.delete("1.0", tk.END)
            self.description_text.insert(tk.END, alpha_data.get('description', ''))
            self.expression_var.set(alpha_data.get('expression', ''))
            self.created_var.set(alpha_data.get('created_at', ''))
            self.modified_var.set(alpha_data.get('updated_at', ''))
            self.tags_var.set(', '.join(alpha_data.get('tags', [])))
            self.status_var.set(alpha_data.get('status', 'Draft'))
            
            # Update editor tab
            self.editor_text.delete("1.0", tk.END)
            self.editor_text.insert(tk.END, alpha_data.get('expression', ''))
            
        except Exception as e:
            logger.error(f"Error loading alpha details: {e}")
            messagebox.showerror("Error", f"Failed to load alpha details: {e}")
    
    def save_alpha(self):
        """Save the current alpha details"""
        alpha_name = self.name_var.get().strip()
        
        if not alpha_name:
            messagebox.showwarning("Missing Name", "Please enter a name for the alpha")
            return
        
        expression = self.expression_var.get().strip()
        if not expression:
            messagebox.showwarning("Missing Expression", "Please enter an expression for the alpha")
            return
        
        # Create alpha data
        alpha_data = {
            'name': alpha_name,
            'description': self.description_text.get("1.0", tk.END).strip(),
            'expression': expression,
            'tags': [tag.strip() for tag in self.tags_var.get().split(',') if tag.strip()],
            'status': self.status_var.get() or 'Draft'
        }
        
        # Add timestamps
        now = datetime.now().isoformat()
        if not self.created_var.get():
            alpha_data['created_at'] = now
        else:
            alpha_data['created_at'] = self.created_var.get()
        alpha_data['updated_at'] = now
        
        # Save to file
        try:
            alpha_file = os.path.join(Config.ALPHA_DIR, f"{alpha_name}.json")
            os.makedirs(os.path.dirname(alpha_file), exist_ok=True)
            
            with open(alpha_file, 'w') as f:
                json.dump(alpha_data, f, indent=2)
            
            # Update list if needed
            if alpha_name not in self.alpha_listbox.get(0, tk.END):
                self.alpha_listbox.insert(tk.END, alpha_name)
                # Select the new item
                index = list(self.alpha_listbox.get(0, tk.END)).index(alpha_name)
                self.alpha_listbox.selection_set(index)
            
            # Update UI
            self.status_var.set(alpha_data['status'])
            self.created_var.set(alpha_data['created_at'])
            self.modified_var.set(alpha_data['updated_at'])
            
            # Update status
            self.app.set_status(f"Alpha '{alpha_name}' saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving alpha: {e}")
            messagebox.showerror("Error", f"Failed to save alpha: {e}")
    
    def generate_alpha(self):
        """Generate a new alpha using AI"""
        if self.alpha_being_generated:
            messagebox.showinfo("In Progress", "Alpha generation is already in progress")
            return
        
        # Start alpha generation process
        self.alpha_being_generated = True
        
        # Create a new alpha entry
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_alpha_name = f"Generated_Alpha_{timestamp}"
        
        # Create simple default alpha
        alpha_data = {
            'name': new_alpha_name,
            'description': "AI Generated Alpha",
            'expression': "rank(close/delay(close, 1)) - rank(volume/delay(volume, 1))",
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
            'tags': ["generated", "momentum", "volume"],
            'status': 'Draft'
        }
        
        # Save to file
        try:
            alpha_file = os.path.join(Config.ALPHA_DIR, f"{new_alpha_name}.json")
            os.makedirs(os.path.dirname(alpha_file), exist_ok=True)
            
            with open(alpha_file, 'w') as f:
                json.dump(alpha_data, f, indent=2)
            
            # Add to list and select
            self.alpha_listbox.insert(tk.END, new_alpha_name)
            index = self.alpha_listbox.get(0, tk.END).index(new_alpha_name)
            self.alpha_listbox.selection_set(index)
            self.load_alpha_details(new_alpha_name)
            
            self.app.set_status(f"New alpha '{new_alpha_name}' generated")
            
        except Exception as e:
            logger.error(f"Error generating alpha: {e}")
            messagebox.showerror("Error", f"Failed to generate alpha: {e}")
        
        finally:
            self.alpha_being_generated = False
    
    def delete_alpha(self):
        """Delete the selected alpha"""
        selection = self.alpha_listbox.curselection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select an alpha to delete")
            return
        
        alpha_name = self.alpha_listbox.get(selection[0])
        
        # Confirm deletion
        if not messagebox.askyesno("Confirm Delete", f"Are you sure you want to delete alpha '{alpha_name}'?"):
            return
        
        # Delete file
        try:
            alpha_file = os.path.join(Config.ALPHA_DIR, f"{alpha_name}.json")
            if os.path.exists(alpha_file):
                os.remove(alpha_file)
            
            # Remove from list
            self.alpha_listbox.delete(selection[0])
            
            # Clear fields
            self.name_var.set("")
            self.description_text.delete("1.0", tk.END)
            self.expression_var.set("")
            self.created_var.set("")
            self.modified_var.set("")
            self.tags_var.set("")
            self.status_var.set("")
            self.editor_text.delete("1.0", tk.END)
            
            self.app.set_status(f"Alpha '{alpha_name}' deleted")
            
        except Exception as e:
            logger.error(f"Error deleting alpha: {e}")
            messagebox.showerror("Error", f"Failed to delete alpha: {e}")
    
    def validate_expression(self):
        """Validate the alpha expression in the editor"""
        expression = self.editor_text.get("1.0", tk.END).strip()
        
        if not expression:
            messagebox.showwarning("Empty Expression", "Please enter an expression to validate")
            return
        
        # This is a placeholder for actual validation logic
        # In a real implementation, this would check syntax and semantics
        valid = True
        
        if valid:
            messagebox.showinfo("Validation", "Alpha expression is valid")
        else:
            messagebox.showerror("Validation Error", "Alpha expression contains errors")
    
    def apply_editor_changes(self):
        """Apply changes from the editor to the alpha details"""
        expression = self.editor_text.get("1.0", tk.END).strip()
        self.expression_var.set(expression)
        messagebox.showinfo("Changes Applied", "Changes have been applied to the alpha expression. Remember to save the alpha.")
    
    def reset_editor(self):
        """Reset the editor to match the current expression"""
        self.editor_text.delete("1.0", tk.END)
        self.editor_text.insert(tk.END, self.expression_var.get())
    
    def backtest_alpha(self):
        """Backtest the selected alpha"""
        selection = self.alpha_listbox.curselection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select an alpha to backtest")
            return
        
        # Switch to backtest tab
        self.alpha_notebook.select(self.backtest_frame)
    
    def run_backtest(self):
        """Run backtest with current settings"""
        selection = self.alpha_listbox.curselection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select an alpha to backtest")
            return
        
        alpha_name = self.alpha_listbox.get(selection[0])
        start_date = self.start_date_var.get()
        end_date = self.end_date_var.get()
        universe = self.universe_var.get()
        
        # Show running status
        self.app.set_status(f"Running backtest for '{alpha_name}'...")
        
        # This would normally run in a separate thread
        # For now, we'll just update the stats with placeholder data
        threading.Thread(target=self._run_backtest_thread, 
                         args=(alpha_name, start_date, end_date, universe)).start()
    
    def _run_backtest_thread(self, alpha_name, start_date, end_date, universe):
        """Background thread for running backtest"""
        try:
            # Simulate backtest running
            import time
            time.sleep(2)
            
            # Update stats text
            stats = f"""
Backtest Results for {alpha_name}
================================
Period: {start_date} to {end_date}
Universe: {universe}

Performance Metrics:
- Annualized Return: 15.2%
- Annualized Volatility: 12.5%
- Sharpe Ratio: 1.22
- Maximum Drawdown: -18.3%
- Information Ratio: 0.87
- Turnover: 42.3%

Factor Exposures:
- Market Beta: 0.05
- Size: -0.12
- Value: 0.33
- Momentum: 0.45
- Volatility: -0.08
            """
            
            # Update UI on main thread
            self.after(0, lambda: self._update_backtest_results(stats))
            self.after(0, lambda: self.app.set_status(f"Backtest completed for '{alpha_name}'"))
            
        except Exception as e:
            logger.error(f"Error in backtest thread: {e}")
            self.after(0, lambda: messagebox.showerror("Backtest Error", f"Error running backtest: {e}"))
            self.after(0, lambda: self.app.set_status("Backtest failed"))
    
    def _update_backtest_results(self, stats):
        """Update the backtest results UI"""
        # Update stats text
        self.stats_text.config(state="normal")
        self.stats_text.delete("1.0", tk.END)
        self.stats_text.insert(tk.END, stats)
        self.stats_text.config(state="disabled") 