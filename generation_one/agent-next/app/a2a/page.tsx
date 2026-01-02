'use client';

import { useState, useEffect, useRef } from 'react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { ScrollArea } from '@/components/ui/scroll-area';
import { A2AService } from '../../lib/a2a-service';
import { Agent, Task, Message } from '../../types/a2a';
import { Loader2, Plus, Edit, Trash2, Save, X } from 'lucide-react';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeRaw from 'rehype-raw';
import rehypeSanitize from 'rehype-sanitize';
import rehypeStringify from 'rehype-stringify';
import mermaid from 'mermaid';

// Initialize Mermaid with dark theme
mermaid.initialize({
  startOnLoad: true,
  theme: 'dark',
  securityLevel: 'loose',
  flowchart: {
    useMaxWidth: true,
    htmlLabels: true,
    curve: 'basis',
  },
  themeVariables: {
    primaryColor: '#3b82f6',
    primaryTextColor: '#fff',
    primaryBorderColor: '#4b5563',
    lineColor: '#6b7280',
    secondaryColor: '#4b5563',
    tertiaryColor: '#6b7280',
    mainBkg: '#1f2937',
    nodeBkg: '#374151',
    nodeBorder: '#4b5563',
    clusterBkg: '#1f2937',
    clusterBorder: '#4b5563',
    titleColor: '#fff',
    edgeLabelBackground: '#1f2937',
    edgeLabelColor: '#e5e7eb',
    labelBoxBkgColor: '#374151',
    labelBoxBorderColor: '#4b5563',
    labelTextColor: '#e5e7eb',
  },
});

// Custom Mermaid component
const Mermaid = ({ chart }: { chart: string }) => {
  const mermaidRef = useRef<HTMLDivElement>(null);
  const [svg, setSvg] = useState<string>('');

  useEffect(() => {
    const renderMermaid = async () => {
      if (mermaidRef.current) {
        try {
          const { svg } = await mermaid.render('mermaid-svg', chart);
          setSvg(svg);
        } catch (error) {
          console.error('Error rendering Mermaid diagram:', error);
        }
      }
    };

    renderMermaid();
  }, [chart]);

  return (
    <div 
      ref={mermaidRef}
      className="mermaid my-8 p-4 bg-gray-900 rounded-lg shadow-lg"
      dangerouslySetInnerHTML={{ __html: svg }}
    />
  );
};

// Initialize service only on client side
const a2aService = typeof window !== 'undefined' ? new A2AService() : null;

const documentation = `
# Agent-to-Agent Protocol

The Agent-to-Agent (A2A) protocol enables autonomous agents to collaborate on quantitative alpha mining tasks.
Each agent has specific capabilities and can communicate with others to achieve complex goals.

## WorldQuant Miner Architecture

\`\`\`mermaid
graph TD
  subgraph Data Collection
    WC[Web Crawler] -->|Financial News| DP[Data Processor]
    FPE[PDF Extractor] -->|Research Papers| DP
    DP -->|Structured Data| DB[(Knowledge Base)]
  end

  subgraph Alpha Research
    AIG[Alpha Idea Generator] -->|New Ideas| AIS[Alpha Idea Simulator]
    AIS -->|Validated Ideas| ACS[Alpha Config Setter]
    ACS -->|Optimized Configs| MAB[Multi-Arm Bandit]
  end

  subgraph Execution
    MAB -->|Best Actions| EX[Executor]
    EX -->|Results| DB
  end

  subgraph Feedback Loop
    DB -->|Historical Data| AIG
    DB -->|Performance Metrics| MAB
  end

  style WC fill:#3b82f6,stroke:#1d4ed8,color:#fff
  style FPE fill:#3b82f6,stroke:#1d4ed8,color:#fff
  style DP fill:#3b82f6,stroke:#1d4ed8,color:#fff
  style AIG fill:#10b981,stroke:#047857,color:#fff
  style AIS fill:#10b981,stroke:#047857,color:#fff
  style ACS fill:#10b981,stroke:#047857,color:#fff
  style MAB fill:#f59e0b,stroke:#b45309,color:#fff
  style EX fill:#f59e0b,stroke:#b45309,color:#fff
  style DB fill:#6366f1,stroke:#4f46e5,color:#fff
\`\`\`

## Prescriptive Agents

### Data Collection Agents
- **Web Crawler Agent**: Specializes in gathering financial news, market data, and research from various online sources
- **PDF Extractor Agent**: Processes and extracts valuable information from financial research papers and reports
- **Data Processor Agent**: Transforms raw data into structured formats and maintains the knowledge base

### Alpha Research Agents
- **Alpha Idea Generator Agent**: Generates new alpha ideas based on market patterns and research insights
- **Alpha Idea Simulator Agent**: Validates and tests alpha ideas through simulation
- **Alpha Config Setter Agent**: Optimizes alpha parameters and configurations

### Execution Agents
- **Multi-Arm Bandit Agent**: Uses reinforcement learning to select the best actions based on historical performance
- **Executor Agent**: Implements selected actions and monitors their performance

## Agent Communication Flow

\`\`\`mermaid
sequenceDiagram
  participant WC as Web Crawler
  participant FPE as PDF Extractor
  participant DP as Data Processor
  participant AIG as Alpha Idea Generator
  participant AIS as Alpha Idea Simulator
  participant ACS as Alpha Config Setter
  participant MAB as Multi-Arm Bandit
  participant EX as Executor
  participant DB as Knowledge Base

  WC->>DP: Financial News
  FPE->>DP: Research Papers
  DP->>DB: Structured Data
  DB->>AIG: Historical Data
  AIG->>AIS: New Alpha Ideas
  AIS->>ACS: Validated Ideas
  ACS->>MAB: Optimized Configs
  MAB->>EX: Best Actions
  EX->>DB: Results
  DB->>MAB: Performance Metrics
\`\`\`

## Features

### Agent Management
- Pre-configured specialized agents for financial research
- Automated workflow orchestration
- Performance monitoring and optimization

### Task Management
- Automated task distribution based on agent capabilities
- Real-time progress tracking
- Priority-based task scheduling

### Communication
- Structured data exchange between agents
- Performance feedback loops
- Automated knowledge sharing

## Getting Started

1. Initialize the knowledge base with historical data
2. Configure agent parameters and capabilities
3. Start the automated research workflow
4. Monitor and analyze results

## Best Practices

- Regularly update the knowledge base with new data
- Monitor agent performance and adjust configurations
- Maintain clear communication protocols between agents
- Implement proper error handling and recovery mechanisms
`;

export default function A2APage() {
  const [agents, setAgents] = useState<Agent[]>([]);
  const [tasks, setTasks] = useState<Task[]>([]);
  const [messages, setMessages] = useState<Message[]>([]);
  const [selectedAgent, setSelectedAgent] = useState<string>('');
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isEditing, setIsEditing] = useState(false);
  const [editingAgent, setEditingAgent] = useState<Partial<Agent> | null>(null);
  const [isCreating, setIsCreating] = useState(false);
  const [newAgent, setNewAgent] = useState<Partial<Agent>>({
    name: '',
    role: '',
    capabilities: [],
    state: { status: 'idle', currentTask: undefined, lastActivity: Date.now() }
  });

  useEffect(() => {
    const initializeData = async () => {
      if (!a2aService) {
        setError('Service not available');
        setIsLoading(false);
        return;
      }

      try {
        setIsLoading(true);
        const [agentsData, tasksData, messagesData] = await Promise.all([
          a2aService.getAgents(),
          a2aService.getTasks(),
          a2aService.getMessages('all')
        ]);
        
        setAgents(agentsData || []);
        setTasks(tasksData || []);
        setMessages(messagesData || []);
        setError(null);
      } catch (error) {
        console.error('Error initializing data:', error);
        setError('Failed to load data');
        setAgents([]);
        setTasks([]);
        setMessages([]);
      } finally {
        setIsLoading(false);
      }
    };

    initializeData();
  }, []);

  useEffect(() => {
    // Re-render Mermaid diagrams when the component mounts
    mermaid.contentLoaded();
  }, []);

  const handleCreateAgent = async () => {
    if (!a2aService) return;

    try {
      const agent: Agent = {
        id: Date.now().toString(),
        name: newAgent.name || '',
        role: newAgent.role || '',
        capabilities: newAgent.capabilities || [],
        state: newAgent.state || { status: 'idle', currentTask: undefined, lastActivity: Date.now() }
      };

      await a2aService.updateAgent(agent);
      const updatedAgents = await a2aService.getAgents();
      setAgents(updatedAgents || []);
      setIsCreating(false);
      setNewAgent({
        name: '',
        role: '',
        capabilities: [],
        state: { status: 'idle', currentTask: undefined, lastActivity: Date.now() }
      });
    } catch (error) {
      console.error('Error creating agent:', error);
      setError('Failed to create agent');
    }
  };

  const handleUpdateAgent = async () => {
    if (!a2aService || !editingAgent?.id) return;

    try {
      const agent = agents.find(a => a.id === editingAgent.id);
      if (!agent) return;

      const updatedAgent = {
        ...agent,
        ...editingAgent
      };

      await a2aService.updateAgent(updatedAgent);
      const updatedAgents = await a2aService.getAgents();
      setAgents(updatedAgents || []);
      setIsEditing(false);
      setEditingAgent(null);
    } catch (error) {
      console.error('Error updating agent:', error);
      setError('Failed to update agent');
    }
  };

  const handleDeleteAgent = async (agentId: string) => {
    if (!a2aService) return;

    try {
      // Note: In a real implementation, you would need to handle tasks and messages
      // associated with this agent before deletion
      const updatedAgents = agents.filter(a => a.id !== agentId);
      setAgents(updatedAgents);
    } catch (error) {
      console.error('Error deleting agent:', error);
      setError('Failed to delete agent');
    }
  };

  const handleTaskAssignment = async (task: Task) => {
    if (!a2aService) return;

    try {
      await a2aService.assignTask(task);
      const updatedTasks = await a2aService.getTasks();
      setTasks(updatedTasks || []);
    } catch (error) {
      console.error('Error assigning task:', error);
      setError('Failed to assign task');
    }
  };

  const handleTaskCompletion = async (taskId: string, result: any) => {
    if (!a2aService) return;

    try {
      await a2aService.completeTask(taskId, result);
      const [updatedTasks, updatedMessages] = await Promise.all([
        a2aService.getTasks(),
        a2aService.getMessages('all')
      ]);
      setTasks(updatedTasks || []);
      setMessages(updatedMessages || []);
    } catch (error) {
      console.error('Error completing task:', error);
      setError('Failed to complete task');
    }
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <Loader2 className="h-8 w-8 animate-spin text-primary" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <h2 className="text-xl font-semibold text-red-500 mb-2">Error</h2>
          <p className="text-muted-foreground">{error}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="container mx-auto p-4">
      <div className="mb-8">
        <h1 className="text-4xl font-bold mb-6 bg-gradient-to-r from-purple-600 via-pink-600 to-blue-600 bg-clip-text text-transparent">
          Agent-to-Agent Protocol
        </h1>
        
        <div className="prose prose-lg max-w-none dark:prose-invert prose-headings:text-primary prose-a:text-blue-600 dark:prose-a:text-blue-400 prose-strong:text-pink-600 dark:prose-strong:text-pink-400 prose-code:bg-gray-100 dark:prose-code:bg-gray-800 prose-code:rounded prose-code:px-2 prose-code:py-1 prose-code:text-sm prose-pre:bg-gray-100 dark:prose-pre:bg-gray-800 prose-pre:rounded-lg prose-pre:p-4 prose-pre:overflow-x-auto">
          <ReactMarkdown
            remarkPlugins={[remarkGfm]}
            rehypePlugins={[rehypeRaw, rehypeSanitize, rehypeStringify]}
            components={{
              h1: ({ children }) => (
                <h1 className="text-3xl font-bold mb-4 bg-gradient-to-r from-purple-600 via-pink-600 to-blue-600 bg-clip-text text-transparent">
                  {children}
                </h1>
              ),
              h2: ({ children }) => (
                <h2 className="text-2xl font-bold mb-3 bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                  {children}
                </h2>
              ),
              h3: ({ children }) => (
                <h3 className="text-xl font-bold mb-2 bg-gradient-to-r from-pink-600 to-purple-600 bg-clip-text text-transparent">
                  {children}
                </h3>
              ),
              p: ({ children }) => (
                <p className="text-gray-700 dark:text-gray-300 mb-4 leading-relaxed">
                  {children}
                </p>
              ),
              ul: ({ children }) => (
                <ul className="list-disc pl-6 mb-4 space-y-2 text-gray-700 dark:text-gray-300">
                  {children}
                </ul>
              ),
              li: ({ children }) => (
                <li className="text-gray-700 dark:text-gray-300">
                  {children}
                </li>
              ),
              code: ({ node, className, children, ...props }) => {
                const match = /language-(\w+)/.exec(className || '');
                if (match && match[1] === 'mermaid') {
                  return <Mermaid chart={String(children).replace(/\n$/, '')} />;
                }
                return (
                  <code className={className} {...props}>
                    {children}
                  </code>
                );
              },
              pre: ({ children }) => (
                <pre className="bg-gray-100 dark:bg-gray-800 rounded-lg p-4 overflow-x-auto my-4">
                  {children}
                </pre>
              ),
            }}
          >
            {documentation}
          </ReactMarkdown>
        </div>
      </div>
      
      <Tabs defaultValue="agents" className="w-full">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="agents">Agents</TabsTrigger>
          <TabsTrigger value="tasks">Tasks</TabsTrigger>
          <TabsTrigger value="messages">Messages</TabsTrigger>
        </TabsList>

        <TabsContent value="agents">
          <div className="flex justify-end mb-4">
            <Dialog open={isCreating} onOpenChange={setIsCreating}>
              <DialogTrigger asChild>
                <Button>
                  <Plus className="h-4 w-4 mr-2" />
                  Add Agent
                </Button>
              </DialogTrigger>
              <DialogContent>
                <DialogHeader>
                  <DialogTitle>Create New Agent</DialogTitle>
                </DialogHeader>
                <div className="space-y-4">
                  <div>
                    <label className="text-sm font-medium">Name</label>
                    <Input
                      value={newAgent.name}
                      onChange={(e) => setNewAgent({ ...newAgent, name: e.target.value })}
                    />
                  </div>
                  <div>
                    <label className="text-sm font-medium">Role</label>
                    <Input
                      value={newAgent.role}
                      onChange={(e) => setNewAgent({ ...newAgent, role: e.target.value })}
                    />
                  </div>
                  <div>
                    <label className="text-sm font-medium">Capabilities</label>
                    <Textarea
                      value={newAgent.capabilities?.join(',')}
                      onChange={(e) => setNewAgent({
                        ...newAgent,
                        capabilities: e.target.value.split(',').map(c => c.trim())
                      })}
                      placeholder="Enter capabilities separated by commas"
                    />
                  </div>
                  <div className="flex justify-end space-x-2">
                    <Button variant="outline" onClick={() => setIsCreating(false)}>
                      Cancel
                    </Button>
                    <Button onClick={handleCreateAgent}>
                      Create
                    </Button>
                  </div>
                </div>
              </DialogContent>
            </Dialog>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {agents.length === 0 ? (
              <div className="col-span-full text-center py-8">
                <p className="text-muted-foreground">No agents available</p>
              </div>
            ) : (
              agents.map((agent) => (
                <Card key={agent.id} className="p-4">
                  {isEditing && editingAgent?.id === agent.id ? (
                    <div className="space-y-4">
                      <Input
                        value={editingAgent.name}
                        onChange={(e) => setEditingAgent({ ...editingAgent, name: e.target.value })}
                      />
                      <Input
                        value={editingAgent.role}
                        onChange={(e) => setEditingAgent({ ...editingAgent, role: e.target.value })}
                      />
                      <Textarea
                        value={editingAgent.capabilities?.join(',')}
                        onChange={(e) => setEditingAgent({
                          ...editingAgent,
                          capabilities: e.target.value.split(',').map(c => c.trim())
                        })}
                      />
                      <div className="flex justify-end space-x-2">
                        <Button variant="outline" onClick={() => setIsEditing(false)}>
                          <X className="h-4 w-4" />
                        </Button>
                        <Button onClick={handleUpdateAgent}>
                          <Save className="h-4 w-4" />
                        </Button>
                      </div>
                    </div>
                  ) : (
                    <>
                      <div className="flex justify-between items-start">
                        <div>
                          <h3 className="font-semibold">{agent.name}</h3>
                          <p className="text-sm text-gray-600">{agent.role}</p>
                        </div>
                        <div className="flex space-x-2">
                          <Button
                            variant="ghost"
                            size="icon"
                            onClick={() => {
                              setEditingAgent(agent);
                              setIsEditing(true);
                            }}
                          >
                            <Edit className="h-4 w-4" />
                          </Button>
                          <Button
                            variant="ghost"
                            size="icon"
                            onClick={() => handleDeleteAgent(agent.id)}
                          >
                            <Trash2 className="h-4 w-4" />
                          </Button>
                        </div>
                      </div>
                      <div className="mt-2">
                        <p className="text-sm">Status: {agent.state.status}</p>
                        <p className="text-sm">Current Task: {agent.state.currentTask || 'None'}</p>
                        <p className="text-sm mt-2">Capabilities:</p>
                        <ul className="text-sm list-disc list-inside">
                          {agent.capabilities.map((cap, i) => (
                            <li key={i}>{cap}</li>
                          ))}
                        </ul>
                      </div>
                    </>
                  )}
                </Card>
              ))
            )}
          </div>
        </TabsContent>

        <TabsContent value="tasks">
          <div className="space-y-4">
            {tasks.length === 0 ? (
              <div className="text-center py-8">
                <p className="text-muted-foreground">No tasks available</p>
              </div>
            ) : (
              tasks.map((task) => (
                <Card key={task.id} className="p-4">
                  <div className="flex justify-between items-start">
                    <div>
                      <h3 className="font-semibold">{task.type}</h3>
                      <p className="text-sm text-gray-600">
                        Status: {task.status}
                      </p>
                      <p className="text-sm">
                        Assigned to: {task.assignedTo || 'Unassigned'}
                      </p>
                    </div>
                    <div className="space-x-2">
                      {task.status === 'pending' && (
                        <Button
                          variant="outline"
                          onClick={() => handleTaskAssignment(task)}
                        >
                          Assign
                        </Button>
                      )}
                      {task.status === 'in_progress' && (
                        <Button
                          variant="outline"
                          onClick={() => handleTaskCompletion(task.id, {})}
                        >
                          Complete
                        </Button>
                      )}
                    </div>
                  </div>
                </Card>
              ))
            )}
          </div>
        </TabsContent>

        <TabsContent value="messages">
          <ScrollArea className="h-[400px]">
            <div className="space-y-4">
              {messages.length === 0 ? (
                <div className="text-center py-8">
                  <p className="text-muted-foreground">No messages available</p>
                </div>
              ) : (
                messages.map((message) => (
                  <Card key={message.id} className="p-4">
                    <div className="flex justify-between items-start">
                      <div>
                        <p className="font-semibold">
                          From: {message.from}
                        </p>
                        <p className="text-sm text-gray-600">
                          To: {message.to}
                        </p>
                        <p className="mt-2">{message.content}</p>
                      </div>
                      <p className="text-sm text-gray-500">
                        {new Date(message.timestamp).toLocaleString()}
                      </p>
                    </div>
                  </Card>
                ))
              )}
            </div>
          </ScrollArea>
        </TabsContent>
      </Tabs>
    </div>
  );
} 