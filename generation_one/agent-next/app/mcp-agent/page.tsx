'use client';

import { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Loader2, Plus, Link, Brain, Settings } from 'lucide-react';
import { useRouter } from 'next/navigation';
import { getStoredCredentials } from '@/lib/auth';
import { openDatabase, getAllFromStore, putInStore } from '@/lib/indexedDB';
import { tools as defaultTools, contexts as defaultContexts } from '@/lib/mcp-data';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';

interface Tool {
  id: string;
  name: string;
  description: string;
  inputSchema: Record<string, any>;
  outputSchema: Record<string, any>;
  result?: any;
  created_at?: number;
  updated_at?: number;
}

interface Context {
  id: string;
  name: string;
  data: any;
  created_at: number;
  updated_at: number;
}

interface LogEntry {
  timestamp: number;
  type: 'info' | 'success' | 'error' | 'warning';
  message: string;
}

interface ToolChain {
  id: string;
  name: string;
  description: string;
  tools: {
    toolId: string;
    inputContext?: string;
    outputContext?: string;
  }[];
  created_at: number;
  updated_at: number;
}

interface SSRNPaper {
  id: number;
  title: string;
  authors: Array<{
    name: string;
    url: string;
  }>;
  abstract: string;
  url: string;
  pdfUrl: string | null;
  publicationStatus: string;
  pageCount: number;
  downloads: number;
  approvedDate: string;
}

interface SSRNResponse {
  total: number;
  papers: SSRNPaper[];
}

export default function MCPAgentPage() {
  const router = useRouter();
  const [isAuthenticated, setIsAuthenticated] = useState<boolean | null>(null);
  const [tools, setTools] = useState<Tool[]>([]);
  const [contexts, setContexts] = useState<Context[]>([]);
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [worker, setWorker] = useState<Worker | null>(null);
  const [db, setDb] = useState<IDBDatabase | null>(null);
  const [isInitialized, setIsInitialized] = useState(false);
  const [toolChains, setToolChains] = useState<ToolChain[]>([]);
  const [showAddToolDialog, setShowAddToolDialog] = useState(false);
  const [showAddChainDialog, setShowAddChainDialog] = useState(false);
  const [newTool, setNewTool] = useState<Partial<Tool>>({});
  const [newChain, setNewChain] = useState<Partial<ToolChain>>({});

  // Check if user is already authenticated
  useEffect(() => {
    const credentials = getStoredCredentials();
    if (credentials) {
      setIsAuthenticated(true);
      setupDatabase();
      setupWorker();
    } else {
      router.push('/login');
    }
  }, [router]);

  const setupDatabase = async () => {
    try {
      const database = await openDatabase();
      setDb(database);
      
      // Initialize tools and contexts if they don't exist
      const existingTools = await getAllFromStore(database, 'tools');
      const existingContexts = await getAllFromStore(database, 'contexts');
      
      if (existingTools.length === 0) {
        // Initialize default tools
        for (const tool of defaultTools) {
          await putInStore(database, 'tools', {
            ...tool,
            created_at: Date.now(),
            updated_at: Date.now()
          });
        }
      }
      
      if (existingContexts.length === 0) {
        // Initialize default contexts
        for (const context of defaultContexts) {
          await putInStore(database, 'contexts', {
            ...context,
            created_at: Date.now(),
            updated_at: Date.now()
          });
        }
      }
      
      await loadToolsAndContexts(database);
      addLog('info', 'Database initialized successfully');
      addLog('info', `Loaded ${tools.length} tools and ${contexts.length} contexts`);
    } catch (error) {
      console.error('Error setting up database:', error);
      addLog('error', 'Failed to setup database');
    }
  };

  const setupWorker = () => {
    const worker = new Worker(new URL('@/workers/mcp-worker.ts', import.meta.url));
    worker.onmessage = (event) => {
      const { type, data } = event.data;
      switch (type) {
        case 'log':
          addLog(data.type, data.message);
          break;
        case 'tool_result':
          handleToolResult(data);
          break;
        case 'context_update':
          handleContextUpdate(data);
          break;
      }
    };
    setWorker(worker);
    addLog('info', 'Web worker initialized');
  };

  const loadToolsAndContexts = async (database: IDBDatabase) => {
    try {
      const loadedTools = await getAllFromStore(database, 'tools');
      const loadedContexts = await getAllFromStore(database, 'contexts');
      
      setTools(loadedTools);
      setContexts(loadedContexts);
      
      addLog('success', `Loaded ${loadedTools.length} tools and ${loadedContexts.length} contexts`);
    } catch (error) {
      console.error('Error loading tools and contexts:', error);
      addLog('error', 'Failed to load tools and contexts');
    }
  };

  const addLog = (type: LogEntry['type'], message: string | { type: string; message: string; data?: any }) => {
    const entry: LogEntry = {
      timestamp: Date.now(),
      type: typeof message === 'string' ? type : message.type as LogEntry['type'],
      message: typeof message === 'string' ? message : message.message
    };
    setLogs(prev => [...prev, entry]);
  };

  const handleToolResult = (data: any) => {
    // Handle tool execution results
    console.log('Tool result:', data);
  };

  const handleContextUpdate = (data: any) => {
    // Handle context updates
    console.log('Context update:', data);
  };

  const executeTool = async (tool: Tool, input: Record<string, any>) => {
    if (!worker) {
      throw new Error('Worker not initialized');
    }

    try {
      // For SSRN crawler, ensure URL is provided
      if (tool.name === 'ssrn_crawler' && !input.url) {
        throw new Error('URL is required for SSRN crawler');
      }

      const result = await new Promise<any>((resolve, reject) => {
        const messageHandler = (event: MessageEvent) => {
          if (event.data.type === 'tool_result' && event.data.data.toolId === tool.id) {
            worker?.removeEventListener('message', messageHandler);
            resolve(event.data.data.result);
          } else if (event.data.type === 'error') {
            worker?.removeEventListener('message', messageHandler);
            reject(new Error(event.data.data.message));
          }
        };

        worker.addEventListener('message', messageHandler);
        worker.postMessage({
          type: 'execute_tool',
          data: {
            toolId: tool.id,
            input
          }
        });
      });

      // Update the tool with the result
      const updatedTools = tools.map(t => {
        if (t.id === tool.id) {
          return { ...t, result };
        }
        return t;
      });
      setTools(updatedTools);

      // Add to logs
      addLog('info', {
        type: 'info',
        message: `Executed ${tool.name}`,
        data: result
      });

      return result;
    } catch (error) {
      console.error('Error executing tool:', error);
      addLog('error', {
        type: 'error',
        message: `Failed to execute ${tool.name}: ${error instanceof Error ? error.message : 'Unknown error'}`
      });
      throw error;
    }
  };

  const initializeToolsAndContexts = async () => {
    try {
      setIsLoading(true);
      addLog('info', 'Starting initialization...');
      
      if (!worker) {
        throw new Error('Web worker not initialized');
      }

      // Send initialization message to worker
      worker.postMessage({
        type: 'initialize',
        data: {
          tools: defaultTools,
          contexts: defaultContexts
        }
      });

      // Wait for initialization complete message
      const handleInitializationComplete = (event: MessageEvent) => {
        if (event.data.type === 'initialization_complete') {
          worker.removeEventListener('message', handleInitializationComplete);
          addLog('success', 'Tools and contexts initialized successfully');
          setIsInitialized(true);
          setIsLoading(false);
        }
      };

      worker.addEventListener('message', handleInitializationComplete);
    } catch (error) {
      console.error('Error during initialization:', error);
      addLog('error', 'Failed to initialize tools and contexts');
      setIsLoading(false);
    }
  };

  const addTool = async () => {
    try {
      if (!db) throw new Error('Database not initialized');
      
      const tool: Tool = {
        id: Date.now().toString(),
        name: newTool.name || '',
        description: newTool.description || '',
        inputSchema: newTool.inputSchema || {},
        outputSchema: newTool.outputSchema || {},
        created_at: Date.now(),
        updated_at: Date.now()
      };

      await putInStore(db, 'tools', tool);
      setTools(prev => [...prev, tool]);
      setShowAddToolDialog(false);
      setNewTool({});
      addLog('success', `Added new tool: ${tool.name}`);
    } catch (error) {
      console.error('Error adding tool:', error);
      addLog('error', 'Failed to add tool');
    }
  };

  const addToolChain = async () => {
    try {
      if (!db) throw new Error('Database not initialized');
      
      const chain: ToolChain = {
        id: Date.now().toString(),
        name: newChain.name || '',
        description: newChain.description || '',
        tools: newChain.tools || [],
        created_at: Date.now(),
        updated_at: Date.now()
      };

      await putInStore(db, 'tool_chains', chain);
      setToolChains(prev => [...prev, chain]);
      setShowAddChainDialog(false);
      setNewChain({});
      addLog('success', `Added new tool chain: ${chain.name}`);
    } catch (error) {
      console.error('Error adding tool chain:', error);
      addLog('error', 'Failed to add tool chain');
    }
  };

  const executeToolChain = async (chainId: string) => {
    try {
      if (!worker || !db) throw new Error('Web worker or database not initialized');
      
      const chain = toolChains.find(c => c.id === chainId);
      if (!chain) throw new Error('Tool chain not found');

      addLog('info', `Executing tool chain: ${chain.name}`);
      
      for (const step of chain.tools) {
        const tool = tools.find(t => t.id === step.toolId);
        if (!tool) continue;

        addLog('info', `Executing tool: ${tool.name}`);
        
        // Get input context if specified
        let inputData: Record<string, any> = {};
        if (step.inputContext) {
          const context = contexts.find(c => c.id === step.inputContext);
          if (context) {
            inputData = { ...context.data };
          }
        }

        // Execute tool
        await executeTool(tool, inputData);

        addLog('success', `Completed tool: ${tool.name}`);
      }

      addLog('success', `Completed tool chain: ${chain.name}`);
    } catch (error) {
      console.error('Error executing tool chain:', error);
      addLog('error', 'Failed to execute tool chain');
    }
  };

  const SSRNPaperList = ({ papers }: { papers: SSRNPaper[] }) => {
    return (
      <div className="space-y-4">
        {papers.map((paper) => (
          <div key={paper.id} className="border rounded-lg p-4">
            <h3 className="text-lg font-semibold">{paper.title}</h3>
            <div className="text-sm text-gray-600">
              {paper.authors.map((author, index) => (
                <span key={author.url}>
                  {author.name}
                  {index < paper.authors.length - 1 ? ', ' : ''}
                </span>
              ))}
            </div>
            <div className="mt-2 text-sm">
              <p>Status: {paper.publicationStatus}</p>
              <p>Pages: {paper.pageCount}</p>
              <p>Downloads: {paper.downloads}</p>
              <p>Approved: {paper.approvedDate}</p>
            </div>
            <div className="mt-4 flex space-x-4">
              <a
                href={paper.url}
                target="_blank"
                rel="noopener noreferrer"
                className="text-blue-600 hover:underline"
              >
                View Paper
              </a>
              {paper.pdfUrl && (
                <a
                  href={paper.pdfUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-blue-600 hover:underline"
                >
                  Download PDF
                </a>
              )}
            </div>
          </div>
        ))}
      </div>
    );
  };

  const ToolResult = ({ tool }: { tool: Tool }) => {
    if (!tool.result) return null;

    switch (tool.name) {
      case 'ssrn_crawler':
        return <SSRNPaperList papers={tool.result.papers} />;
      default:
        return <pre className="whitespace-pre-wrap">{JSON.stringify(tool.result, null, 2)}</pre>;
    }
  };

  const handleToolExecution = (tool: Tool) => {
    // For SSRN crawler, use the default API URL
    const input = tool.name === 'ssrn_crawler' 
      ? { url: 'https://api.ssrn.com/content/v1/bindings/2978227/papers?index=0&count=50&sort=0' }
      : tool.inputSchema;
    
    executeTool(tool, input);
  };

  if (isAuthenticated === null) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <Loader2 className="h-8 w-8 animate-spin text-primary" />
      </div>
    );
  }

  if (!isAuthenticated) {
    return null;
  }

  return (
    <div className="container mx-auto py-4 space-y-4">
      <div className="flex flex-col space-y-1">
        <h1 className="text-2xl font-bold">MCP Agent</h1>
        <p className="text-sm text-muted-foreground">
          Model Context Protocol Agent with Web Worker
        </p>
      </div>

      {!isInitialized && (
        <Card>
          <CardHeader>
            <CardTitle>Initialization Required</CardTitle>
            <CardDescription>Please initialize the MCP agent to get started</CardDescription>
          </CardHeader>
          <CardContent>
            <Button 
              onClick={initializeToolsAndContexts}
              disabled={isLoading}
            >
              {isLoading ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Initializing...
                </>
              ) : (
                'Initialize MCP Agent'
              )}
            </Button>
          </CardContent>
        </Card>
      )}

      {isInitialized && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          {/* Tools Section */}
          <Card>
            <CardHeader className="flex flex-row items-center justify-between">
              <div>
                <CardTitle>Tools</CardTitle>
                <CardDescription>Available tools for the agent</CardDescription>
              </div>
              <Dialog open={showAddToolDialog} onOpenChange={setShowAddToolDialog}>
                <DialogTrigger asChild>
                  <Button variant="outline" size="sm">
                    <Plus className="h-4 w-4 mr-2" />
                    Add Tool
                  </Button>
                </DialogTrigger>
                <DialogContent>
                  <DialogHeader>
                    <DialogTitle>Add New Tool</DialogTitle>
                  </DialogHeader>
                  <div className="space-y-4">
                    <Input
                      placeholder="Tool Name"
                      value={newTool.name || ''}
                      onChange={(e) => setNewTool({ ...newTool, name: e.target.value })}
                    />
                    <Textarea
                      placeholder="Tool Description"
                      value={newTool.description || ''}
                      onChange={(e) => setNewTool({ ...newTool, description: e.target.value })}
                    />
                    <Textarea
                      placeholder="Tool Input Schema"
                      value={JSON.stringify(newTool.inputSchema || {}, null, 2)}
                      onChange={(e) => setNewTool({ ...newTool, inputSchema: JSON.parse(e.target.value) })}
                    />
                    <Textarea
                      placeholder="Tool Output Schema"
                      value={JSON.stringify(newTool.outputSchema || {}, null, 2)}
                      onChange={(e) => setNewTool({ ...newTool, outputSchema: JSON.parse(e.target.value) })}
                    />
                    <Button onClick={addTool}>Add Tool</Button>
                  </div>
                </DialogContent>
              </Dialog>
            </CardHeader>
            <CardContent>
              <ScrollArea className="h-[300px]">
                <div className="space-y-2">
                  {tools.map(tool => (
                    <div key={tool.id} className="p-2 border rounded">
                      <h3 className="font-medium">{tool.name}</h3>
                      <p className="text-sm text-muted-foreground">{tool.description}</p>
                      <Button 
                        variant="outline" 
                        size="sm" 
                        className="mt-2"
                        onClick={() => handleToolExecution(tool)}
                      >
                        Execute
                      </Button>
                    </div>
                  ))}
                </div>
              </ScrollArea>
            </CardContent>
          </Card>

          {/* Contexts Section */}
          <Card>
            <CardHeader>
              <CardTitle>Contexts</CardTitle>
              <CardDescription>Current context data</CardDescription>
            </CardHeader>
            <CardContent>
              <ScrollArea className="h-[300px]">
                <div className="space-y-2">
                  {contexts.map(context => (
                    <div key={context.id} className="p-2 border rounded">
                      <h3 className="font-medium">{context.name}</h3>
                      <pre className="text-sm text-muted-foreground overflow-x-auto">
                        {JSON.stringify(context.data, null, 2)}
                      </pre>
                    </div>
                  ))}
                </div>
              </ScrollArea>
            </CardContent>
          </Card>

          {/* Tool Chains Section */}
          <Card>
            <CardHeader className="flex flex-row items-center justify-between">
              <div>
                <CardTitle>Tool Chains</CardTitle>
                <CardDescription>Chained tool executions</CardDescription>
              </div>
              <Dialog open={showAddChainDialog} onOpenChange={setShowAddChainDialog}>
                <DialogTrigger asChild>
                  <Button variant="outline" size="sm">
                    <Link className="h-4 w-4 mr-2" />
                    Add Chain
                  </Button>
                </DialogTrigger>
                <DialogContent>
                  <DialogHeader>
                    <DialogTitle>Add New Tool Chain</DialogTitle>
                  </DialogHeader>
                  <div className="space-y-4">
                    <Input
                      placeholder="Chain Name"
                      value={newChain.name || ''}
                      onChange={(e) => setNewChain({ ...newChain, name: e.target.value })}
                    />
                    <Textarea
                      placeholder="Chain Description"
                      value={newChain.description || ''}
                      onChange={(e) => setNewChain({ ...newChain, description: e.target.value })}
                    />
                    <div className="space-y-2">
                      {newChain.tools?.map((step, index) => (
                        <div key={index} className="flex space-x-2">
                          <Select
                            value={step.toolId}
                            onValueChange={(value) => {
                              const newTools = [...(newChain.tools || [])];
                              newTools[index].toolId = value;
                              setNewChain({ ...newChain, tools: newTools });
                            }}
                          >
                            <SelectTrigger>
                              <SelectValue placeholder="Select Tool" />
                            </SelectTrigger>
                            <SelectContent>
                              {tools.map(tool => (
                                <SelectItem key={tool.id} value={tool.id}>
                                  {tool.name}
                                </SelectItem>
                              ))}
                            </SelectContent>
                          </Select>
                          <Select
                            value={step.inputContext}
                            onValueChange={(value) => {
                              const newTools = [...(newChain.tools || [])];
                              newTools[index].inputContext = value;
                              setNewChain({ ...newChain, tools: newTools });
                            }}
                          >
                            <SelectTrigger>
                              <SelectValue placeholder="Input Context" />
                            </SelectTrigger>
                            <SelectContent>
                              {contexts.map(context => (
                                <SelectItem key={context.id} value={context.id}>
                                  {context.name}
                                </SelectItem>
                              ))}
                            </SelectContent>
                          </Select>
                          <Select
                            value={step.outputContext}
                            onValueChange={(value) => {
                              const newTools = [...(newChain.tools || [])];
                              newTools[index].outputContext = value;
                              setNewChain({ ...newChain, tools: newTools });
                            }}
                          >
                            <SelectTrigger>
                              <SelectValue placeholder="Output Context" />
                            </SelectTrigger>
                            <SelectContent>
                              {contexts.map(context => (
                                <SelectItem key={context.id} value={context.id}>
                                  {context.name}
                                </SelectItem>
                              ))}
                            </SelectContent>
                          </Select>
                        </div>
                      ))}
                      <Button
                        variant="outline"
                        onClick={() => {
                          setNewChain({
                            ...newChain,
                            tools: [...(newChain.tools || []), { toolId: '', inputContext: '', outputContext: '' }]
                          });
                        }}
                      >
                        Add Step
                      </Button>
                    </div>
                    <Button onClick={addToolChain}>Add Chain</Button>
                  </div>
                </DialogContent>
              </Dialog>
            </CardHeader>
            <CardContent>
              <ScrollArea className="h-[300px]">
                <div className="space-y-2">
                  {toolChains.map(chain => (
                    <div key={chain.id} className="p-2 border rounded">
                      <h3 className="font-medium">{chain.name}</h3>
                      <p className="text-sm text-muted-foreground">{chain.description}</p>
                      <div className="mt-2 space-y-1">
                        {chain.tools.map((step, index) => {
                          const tool = tools.find(t => t.id === step.toolId);
                          return (
                            <div key={index} className="text-sm">
                              {index + 1}. {tool?.name}
                              {step.inputContext && ` ← ${contexts.find(c => c.id === step.inputContext)?.name}`}
                              {step.outputContext && ` → ${contexts.find(c => c.id === step.outputContext)?.name}`}
                            </div>
                          );
                        })}
                      </div>
                      <Button
                        variant="outline"
                        size="sm"
                        className="mt-2"
                        onClick={() => executeToolChain(chain.id)}
                      >
                        Execute Chain
                      </Button>
                    </div>
                  ))}
                </div>
              </ScrollArea>
            </CardContent>
          </Card>

          {/* Logs Section */}
          <Card className="col-span-3">
            <CardHeader>
              <CardTitle>Logs</CardTitle>
              <CardDescription>Agent activity log</CardDescription>
            </CardHeader>
            <CardContent>
              <ScrollArea className="h-[300px]">
                <div className="space-y-1">
                  {logs.map((log, index) => (
                    <div key={index} className="flex items-start">
                      <span className="text-gray-500 mr-1 text-xs">
                        {new Date(log.timestamp).toLocaleTimeString()}
                      </span>
                      <span className={`text-xs ${
                        log.type === 'info' ? 'text-blue-400' :
                        log.type === 'success' ? 'text-green-400' :
                        log.type === 'error' ? 'text-red-400' :
                        'text-yellow-400'
                      }`}>
                        {log.message}
                      </span>
                    </div>
                  ))}
                </div>
              </ScrollArea>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
} 