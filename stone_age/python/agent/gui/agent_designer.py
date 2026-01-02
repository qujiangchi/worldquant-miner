#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Agent Designer tab for the Alpha Agent Network application.
Allows users to create, configure and manage AI agents.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import json
import os
import uuid
import logging
from datetime import datetime

# Application imports
from agent.config import Config

logger = logging.getLogger(__name__)

class AgentDesignerFrame(ttk.Frame):
    """Frame for the Agent Designer tab"""
    
    def __init__(self, parent, app):
        """Initialize the frame"""
        super().__init__(parent)
        self.app = app
        self.agents = {}
        self.current_agent_id = None
        
        self.setup_ui()
        self.load_agents()
    
    def setup_ui(self):
        """Set up the UI components"""
        # Split into left (agent list) and right (agent details) frames
        self.paned_window = ttk.PanedWindow(self, orient="horizontal")
        self.paned_window.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Left frame - Agent list and control buttons
        self.left_frame = ttk.Frame(self.paned_window)
        self.paned_window.add(self.left_frame, weight=1)
        
        # Right frame - Agent configuration
        self.right_frame = ttk.Frame(self.paned_window)
        self.paned_window.add(self.right_frame, weight=2)
        
        # Set up the left frame (agent list)
        self.setup_agent_list()
        
        # Set up the right frame (agent configuration)
        self.setup_agent_config()
    
    def setup_agent_list(self):
        """Set up the agent list panel"""
        # Title label
        ttk.Label(self.left_frame, text="Available Agents", font=("TkDefaultFont", 12, "bold")).pack(anchor="w", padx=5, pady=5)
        
        # Agent list
        list_frame = ttk.Frame(self.left_frame)
        list_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Create the Treeview for agents
        self.agent_tree = ttk.Treeview(list_frame, columns=("name", "type", "status"), show="headings", height=15)
        self.agent_tree.heading("name", text="Name")
        self.agent_tree.heading("type", text="Type")
        self.agent_tree.heading("status", text="Status")
        
        self.agent_tree.column("name", width=150)
        self.agent_tree.column("type", width=100)
        self.agent_tree.column("status", width=80)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.agent_tree.yview)
        self.agent_tree.configure(yscrollcommand=scrollbar.set)
        
        self.agent_tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Bind selection event
        self.agent_tree.bind("<<TreeviewSelect>>", self.on_agent_selected)
        
        # Control buttons
        button_frame = ttk.Frame(self.left_frame)
        button_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Button(button_frame, text="New Agent", command=self.new_agent).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Clone Agent", command=self.clone_agent).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Delete Agent", command=self.delete_agent).pack(side="left", padx=5)
        
        # Agent control frame
        control_frame = ttk.LabelFrame(self.left_frame, text="Agent Control")
        control_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Button(control_frame, text="Start Agent", command=self.start_agent).pack(side="left", padx=5, pady=5, fill="x", expand=True)
        ttk.Button(control_frame, text="Stop Agent", command=self.stop_agent).pack(side="left", padx=5, pady=5, fill="x", expand=True)
    
    def setup_agent_config(self):
        """Set up the agent configuration panel"""
        # Create a notebook for configuration tabs
        self.config_notebook = ttk.Notebook(self.right_frame)
        self.config_notebook.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Basic configuration tab
        self.basic_config_frame = ttk.Frame(self.config_notebook)
        self.config_notebook.add(self.basic_config_frame, text="Basic Config")
        
        # Capabilities tab
        self.capabilities_frame = ttk.Frame(self.config_notebook)
        self.config_notebook.add(self.capabilities_frame, text="Capabilities")
        
        # Instructions tab
        self.instructions_frame = ttk.Frame(self.config_notebook)
        self.config_notebook.add(self.instructions_frame, text="Instructions")
        
        # Advanced tab
        self.advanced_frame = ttk.Frame(self.config_notebook)
        self.config_notebook.add(self.advanced_frame, text="Advanced")
        
        # Set up the basic configuration tab
        self.setup_basic_config()
        
        # Set up the capabilities tab
        self.setup_capabilities()
        
        # Set up the instructions tab
        self.setup_instructions()
        
        # Set up the advanced tab
        self.setup_advanced_config()
        
        # Save button
        ttk.Button(self.right_frame, text="Save Agent", command=self.save_agent).pack(fill="x", padx=5, pady=5)
    
    def setup_basic_config(self):
        """Set up the basic configuration tab"""
        frame = ttk.Frame(self.basic_config_frame)
        frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Agent name
        ttk.Label(frame, text="Agent Name:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.name_var = tk.StringVar()
        ttk.Entry(frame, textvariable=self.name_var, width=40).grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        
        # Agent type
        ttk.Label(frame, text="Agent Type:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.type_var = tk.StringVar()
        agent_types = ["Research", "Crawler", "Analyst", "Generator", "Validator"]
        ttk.Combobox(frame, textvariable=self.type_var, values=agent_types).grid(row=1, column=1, sticky="ew", padx=5, pady=5)
        
        # Agent description
        ttk.Label(frame, text="Description:").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        self.description_text = scrolledtext.ScrolledText(frame, width=40, height=5)
        self.description_text.grid(row=2, column=1, sticky="ew", padx=5, pady=5)
        
        # Model
        ttk.Label(frame, text="Model:").grid(row=3, column=0, sticky="w", padx=5, pady=5)
        self.model_var = tk.StringVar(value=Config.DEFAULT_MODEL)
        models = ["gpt-4o", "gpt-4-turbo", "claude-3-opus", "claude-3-sonnet", "llama-3-70b"]
        ttk.Combobox(frame, textvariable=self.model_var, values=models).grid(row=3, column=1, sticky="ew", padx=5, pady=5)
        
        # Temperature
        ttk.Label(frame, text="Temperature:").grid(row=4, column=0, sticky="w", padx=5, pady=5)
        temp_frame = ttk.Frame(frame)
        temp_frame.grid(row=4, column=1, sticky="ew", padx=5, pady=5)
        
        self.temperature_var = tk.DoubleVar(value=0.7)
        ttk.Scale(temp_frame, from_=0.0, to=1.0, variable=self.temperature_var, orient="horizontal").pack(side="left", fill="x", expand=True)
        ttk.Label(temp_frame, textvariable=self.temperature_var).pack(side="right", padx=5)
        
        # Add some spacing
        frame.columnconfigure(1, weight=1)
    
    def setup_capabilities(self):
        """Set up the capabilities tab"""
        frame = ttk.Frame(self.capabilities_frame)
        frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Tools frame
        tools_frame = ttk.LabelFrame(frame, text="Agent Tools")
        tools_frame.pack(fill="x", padx=5, pady=5)
        
        # Tool checkboxes
        self.tool_vars = {}
        tools = [
            ("web_search", "Web Search"),
            ("web_browser", "Web Browser"),
            ("file_reader", "File Reader"),
            ("file_writer", "File Writer"),
            ("code_executor", "Code Executor"),
            ("data_analyzer", "Data Analyzer"),
            ("chart_generator", "Chart Generator"),
            ("alpha_tester", "Alpha Tester")
        ]
        
        for i, (tool_id, tool_name) in enumerate(tools):
            self.tool_vars[tool_id] = tk.BooleanVar(value=False)
            ttk.Checkbutton(tools_frame, text=tool_name, variable=self.tool_vars[tool_id]).grid(
                row=i//2, column=i%2, sticky="w", padx=10, pady=5
            )
        
        # Memory configuration
        memory_frame = ttk.LabelFrame(frame, text="Memory Configuration")
        memory_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Label(memory_frame, text="Memory Type:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.memory_type_var = tk.StringVar(value="buffer")
        memory_types = ["buffer", "summary", "vectorstore", "none"]
        ttk.Combobox(memory_frame, textvariable=self.memory_type_var, values=memory_types).grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        
        ttk.Label(memory_frame, text="Memory Size:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.memory_size_var = tk.IntVar(value=10)
        ttk.Spinbox(memory_frame, from_=1, to=100, textvariable=self.memory_size_var).grid(row=1, column=1, sticky="ew", padx=5, pady=5)
    
    def setup_instructions(self):
        """Set up the instructions tab"""
        frame = ttk.Frame(self.instructions_frame)
        frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # System prompt
        ttk.Label(frame, text="System Prompt:").pack(anchor="w", padx=5, pady=5)
        self.system_prompt_text = scrolledtext.ScrolledText(frame, width=50, height=10)
        self.system_prompt_text.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Default system prompts
        templates_frame = ttk.LabelFrame(frame, text="Template Prompts")
        templates_frame.pack(fill="x", padx=5, pady=5)
        
        # Template buttons
        ttk.Button(templates_frame, text="Research Agent", 
                 command=lambda: self.load_template("research")).pack(side="left", padx=5, pady=5)
        ttk.Button(templates_frame, text="Crawler Agent", 
                 command=lambda: self.load_template("crawler")).pack(side="left", padx=5, pady=5)
        ttk.Button(templates_frame, text="Analyst Agent", 
                 command=lambda: self.load_template("analyst")).pack(side="left", padx=5, pady=5)
        ttk.Button(templates_frame, text="Generator Agent", 
                 command=lambda: self.load_template("generator")).pack(side="left", padx=5, pady=5)
    
    def setup_advanced_config(self):
        """Set up the advanced configuration tab"""
        frame = ttk.Frame(self.advanced_frame)
        frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Add a JSON editor for advanced configuration
        ttk.Label(frame, text="Advanced Configuration (JSON):").pack(anchor="w", padx=5, pady=5)
        self.advanced_config_text = scrolledtext.ScrolledText(frame, width=50, height=20)
        self.advanced_config_text.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Default advanced configuration
        self.default_advanced_config = {
            "max_iterations": 10,
            "request_timeout": 60,
            "llm_cache": True,
            "streaming": True,
            "verbose": True,
            "custom_tools": [],
            "allowed_sites": ["*"],
            "blocked_sites": []
        }
        
        # Set default advanced config
        self.advanced_config_text.delete("1.0", tk.END)
        self.advanced_config_text.insert("1.0", json.dumps(self.default_advanced_config, indent=2))
    
    def load_template(self, template_type):
        """Load a template system prompt based on agent type"""
        templates = {
            "research": """You are a Financial Research Agent specialized in analyzing financial markets and trends.
Your goal is to gather and synthesize information about financial instruments, trends, and alpha strategies.
You can search the web, analyze data, and provide insights based on your findings.

Key Responsibilities:
1. Search for academic papers and research articles on quantitative finance
2. Analyze market trends and patterns
3. Identify potential alpha factors from financial literature
4. Summarize findings in a clear, actionable format

Provide research summaries in this format:
- Source: [Source of information]
- Key Findings: [Brief summary of main findings]
- Potential Alpha Factors: [List any potential alpha factors identified]
- Relevance: [Rate relevance for alpha generation on a scale of 1-10]
""",
            "crawler": """You are a Financial Web Crawler Agent specialized in efficiently navigating financial websites.
Your goal is to extract relevant information from financial research sites, blogs, and academic sources.
You can browse websites, follow links, and extract structured data from web pages.

Key Responsibilities:
1. Crawl financial research websites systematically
2. Extract articles, research papers, and blog posts about quantitative finance
3. Identify and extract potential alpha strategies and factors
4. Save extracted information in a structured format for further analysis

When extracting data, focus on:
- Quantitative trading strategies
- Statistical arbitrage approaches
- Factor modeling techniques
- Market anomalies and inefficiencies
- Machine learning applications in finance
""",
            "analyst": """You are a Financial Data Analyst Agent specialized in evaluating alpha factors.
Your goal is to analyze financial data, evaluate alpha strategies, and provide insights on their effectiveness.
You can process data, create visualizations, and calculate key performance metrics.

Key Responsibilities:
1. Analyze historical data to evaluate alpha strategies
2. Calculate key metrics like Sharpe ratio, drawdown, and factor exposure
3. Identify potential improvements to alpha strategies
4. Provide data-backed recommendations for alpha refinement

When analyzing alphas, consider these aspects:
- Statistical significance (t-stat)
- Consistency across market regimes
- Transaction costs and turnover
- Factor exposure and potential crowding
- Correlation with existing factors
""",
            "generator": """You are an Alpha Generation Agent specialized in creating quantitative trading strategies.
Your goal is to formulate alpha expressions that capture market inefficiencies and generate excess returns.
You can generate alpha expressions, explain their rationale, and recommend improvements.

Key Responsibilities:
1. Generate alpha expressions based on research findings
2. Explain the economic rationale behind each alpha
3. Suggest improvements to existing alphas
4. Create diverse alphas across different factors and timeframes

When generating alphas, follow these guidelines:
- Use standard operators (rank, decay, delta, etc.)
- Ensure expressions are computable from available data
- Focus on one clear hypothesis per alpha
- Avoid overly complex expressions prone to overfitting
- Consider implementation constraints like liquidity and turnover
"""
        }
        
        if template_type in templates:
            self.system_prompt_text.delete("1.0", tk.END)
            self.system_prompt_text.insert("1.0", templates[template_type])
    
    def load_agents(self):
        """Load saved agents from disk"""
        agents_dir = Config.AGENTS_DIR
        if not os.path.exists(agents_dir):
            os.makedirs(agents_dir)
            return
        
        # Clear existing entries
        for item in self.agent_tree.get_children():
            self.agent_tree.delete(item)
        
        self.agents = {}
        
        # Load each agent file
        for filename in os.listdir(agents_dir):
            if filename.endswith('.json'):
                try:
                    with open(os.path.join(agents_dir, filename), 'r') as f:
                        agent_data = json.load(f)
                    
                    agent_id = agent_data.get('id', filename.split('.')[0])
                    self.agents[agent_id] = agent_data
                    
                    # Add to treeview
                    status = "Active" if self.app.active_agents.get(agent_id, {}).get('active', False) else "Inactive"
                    self.agent_tree.insert("", "end", agent_id, values=(
                        agent_data.get('name', 'Unnamed'),
                        agent_data.get('type', 'Unknown'),
                        status
                    ))
                    
                except Exception as e:
                    logger.error(f"Error loading agent {filename}: {e}")
    
    def on_agent_selected(self, event):
        """Handle agent selection in the treeview"""
        selected_items = self.agent_tree.selection()
        if not selected_items:
            return
        
        agent_id = selected_items[0]
        if agent_id not in self.agents:
            return
        
        self.current_agent_id = agent_id
        agent_data = self.agents[agent_id]
        
        # Update basic configuration fields
        self.name_var.set(agent_data.get('name', ''))
        self.type_var.set(agent_data.get('type', ''))
        self.description_text.delete("1.0", tk.END)
        self.description_text.insert("1.0", agent_data.get('description', ''))
        self.model_var.set(agent_data.get('model', Config.DEFAULT_MODEL))
        self.temperature_var.set(agent_data.get('temperature', 0.7))
        
        # Update capabilities
        for tool_id in self.tool_vars:
            self.tool_vars[tool_id].set(tool_id in agent_data.get('tools', []))
        
        self.memory_type_var.set(agent_data.get('memory_type', 'buffer'))
        self.memory_size_var.set(agent_data.get('memory_size', 10))
        
        # Update system prompt
        self.system_prompt_text.delete("1.0", tk.END)
        self.system_prompt_text.insert("1.0", agent_data.get('system_prompt', ''))
        
        # Update advanced configuration
        self.advanced_config_text.delete("1.0", tk.END)
        advanced_config = agent_data.get('advanced_config', self.default_advanced_config)
        self.advanced_config_text.insert("1.0", json.dumps(advanced_config, indent=2))
    
    def new_agent(self):
        """Create a new agent"""
        agent_id = str(uuid.uuid4())
        agent_data = {
            'id': agent_id,
            'name': f"New Agent {len(self.agents) + 1}",
            'type': 'Research',
            'description': 'A new agent',
            'model': Config.DEFAULT_MODEL,
            'temperature': 0.7,
            'tools': [],
            'memory_type': 'buffer',
            'memory_size': 10,
            'system_prompt': '',
            'advanced_config': self.default_advanced_config,
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat()
        }
        
        self.agents[agent_id] = agent_data
        self.agent_tree.insert("", "end", agent_id, values=(
            agent_data['name'],
            agent_data['type'],
            "Inactive"
        ))
        
        # Select the new agent
        self.agent_tree.selection_set(agent_id)
        self.on_agent_selected(None)
        
        # Save the new agent
        self.save_agent()
    
    def clone_agent(self):
        """Clone the selected agent"""
        if not self.current_agent_id or self.current_agent_id not in self.agents:
            messagebox.showerror("Error", "No agent selected to clone")
            return
        
        # Create a new agent based on the selected one
        agent_id = str(uuid.uuid4())
        agent_data = self.agents[self.current_agent_id].copy()
        agent_data['id'] = agent_id
        agent_data['name'] = f"{agent_data['name']} (Clone)"
        agent_data['created_at'] = datetime.now().isoformat()
        agent_data['updated_at'] = datetime.now().isoformat()
        
        self.agents[agent_id] = agent_data
        self.agent_tree.insert("", "end", agent_id, values=(
            agent_data['name'],
            agent_data['type'],
            "Inactive"
        ))
        
        # Select the new agent
        self.agent_tree.selection_set(agent_id)
        self.on_agent_selected(None)
        
        # Save the new agent
        self.save_agent()
    
    def delete_agent(self):
        """Delete the selected agent"""
        if not self.current_agent_id:
            messagebox.showerror("Error", "No agent selected to delete")
            return
        
        # Confirm deletion
        if not messagebox.askyesno("Confirm Delete", f"Are you sure you want to delete the agent '{self.agents[self.current_agent_id].get('name', 'Unnamed')}'?"):
            return
        
        # Stop the agent if it's running
        if self.app.active_agents.get(self.current_agent_id, {}).get('active', False):
            self.stop_agent()
        
        # Delete the agent file
        agent_file = os.path.join(Config.AGENTS_DIR, f"{self.current_agent_id}.json")
        if os.path.exists(agent_file):
            try:
                os.remove(agent_file)
            except Exception as e:
                logger.error(f"Error deleting agent file {agent_file}: {e}")
        
        # Remove from tree and dictionary
        self.agent_tree.delete(self.current_agent_id)
        if self.current_agent_id in self.agents:
            del self.agents[self.current_agent_id]
        
        # Clear current selection
        self.current_agent_id = None
    
    def save_agent(self):
        """Save the current agent configuration"""
        if not self.current_agent_id:
            messagebox.showerror("Error", "No agent selected to save")
            return
        
        try:
            # Get values from UI
            agent_data = {
                'id': self.current_agent_id,
                'name': self.name_var.get(),
                'type': self.type_var.get(),
                'description': self.description_text.get("1.0", tk.END).strip(),
                'model': self.model_var.get(),
                'temperature': self.temperature_var.get(),
                'tools': [tool_id for tool_id, var in self.tool_vars.items() if var.get()],
                'memory_type': self.memory_type_var.get(),
                'memory_size': self.memory_size_var.get(),
                'system_prompt': self.system_prompt_text.get("1.0", tk.END).strip(),
                'updated_at': datetime.now().isoformat()
            }
            
            # Parse advanced config JSON
            try:
                advanced_config = json.loads(self.advanced_config_text.get("1.0", tk.END))
                agent_data['advanced_config'] = advanced_config
            except json.JSONDecodeError as e:
                messagebox.showerror("Invalid JSON", f"Advanced configuration contains invalid JSON: {e}")
                return
            
            # Preserve created_at
            if self.current_agent_id in self.agents and 'created_at' in self.agents[self.current_agent_id]:
                agent_data['created_at'] = self.agents[self.current_agent_id]['created_at']
            else:
                agent_data['created_at'] = datetime.now().isoformat()
            
            # Update agents dictionary
            self.agents[self.current_agent_id] = agent_data
            
            # Update treeview
            status = "Active" if self.app.active_agents.get(self.current_agent_id, {}).get('active', False) else "Inactive"
            self.agent_tree.item(self.current_agent_id, values=(
                agent_data['name'],
                agent_data['type'],
                status
            ))
            
            # Save to file
            agent_file = os.path.join(Config.AGENTS_DIR, f"{self.current_agent_id}.json")
            os.makedirs(os.path.dirname(agent_file), exist_ok=True)
            with open(agent_file, 'w') as f:
                json.dump(agent_data, f, indent=2)
            
            # Show success message
            self.app.set_status(f"Agent '{agent_data['name']}' saved successfully")
            
        except Exception as e:
            logger.exception(f"Error saving agent {self.current_agent_id}")
            messagebox.showerror("Error", f"Failed to save agent: {e}")
    
    def start_agent(self):
        """Start the selected agent"""
        if not self.current_agent_id:
            messagebox.showerror("Error", "No agent selected to start")
            return
        
        # Check if already running
        if self.app.active_agents.get(self.current_agent_id, {}).get('active', False):
            messagebox.showinfo("Info", "Agent is already running")
            return
        
        # Save current configuration
        self.save_agent()
        
        try:
            # Create agent instance (placeholder for actual agent creation)
            from agent.core.agent_factory import create_agent
            agent_instance = create_agent(self.agents[self.current_agent_id])
            
            # Add to active agents
            self.app.active_agents[self.current_agent_id] = {
                'instance': agent_instance,
                'active': True,
                'started_at': datetime.now().isoformat()
            }
            
            # Update UI
            self.agent_tree.item(self.current_agent_id, values=(
                self.agents[self.current_agent_id]['name'],
                self.agents[self.current_agent_id]['type'],
                "Active"
            ))
            
            # Update agent count
            self.app.update_agent_count()
            
            # Show success message
            self.app.set_status(f"Agent '{self.agents[self.current_agent_id]['name']}' started")
            
        except Exception as e:
            logger.exception(f"Error starting agent {self.current_agent_id}")
            messagebox.showerror("Error", f"Failed to start agent: {e}")
    
    def stop_agent(self):
        """Stop the selected agent"""
        if not self.current_agent_id:
            messagebox.showerror("Error", "No agent selected to stop")
            return
        
        # Check if running
        if not self.app.active_agents.get(self.current_agent_id, {}).get('active', False):
            messagebox.showinfo("Info", "Agent is not running")
            return
        
        try:
            # Stop agent instance
            agent_info = self.app.active_agents[self.current_agent_id]
            agent_info['instance'].stop()
            agent_info['active'] = False
            
            # Update UI
            self.agent_tree.item(self.current_agent_id, values=(
                self.agents[self.current_agent_id]['name'],
                self.agents[self.current_agent_id]['type'],
                "Inactive"
            ))
            
            # Update agent count
            self.app.update_agent_count()
            
            # Show success message
            self.app.set_status(f"Agent '{self.agents[self.current_agent_id]['name']}' stopped")
            
        except Exception as e:
            logger.exception(f"Error stopping agent {self.current_agent_id}")
            messagebox.showerror("Error", f"Failed to stop agent: {e}") 