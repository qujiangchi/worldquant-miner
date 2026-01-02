import { Agent, Task, Message, AgentCommunication, AgentTaskManagement, AlphaMiningProtocol } from '../types/a2a';
import { createChatCompletion } from './deepseek';

export class A2AService implements AgentCommunication, AgentTaskManagement, AlphaMiningProtocol {
  private db: IDBDatabase | null = null;
  private worker: Worker | null = null;
  private isInitialized = false;
  private messageHandlers: Map<string, ((data: any) => void)[]> = new Map();
  private initializationPromise: Promise<void>;

  constructor() {
    // Only initialize on client side
    if (typeof window !== 'undefined') {
      this.initializationPromise = this.initialize();
    } else {
      this.initializationPromise = Promise.reject(new Error('Not running in browser environment'));
    }
  }

  private async initialize(): Promise<void> {
    try {
      await this.initializeDB();
      this.initializeWorker();
      this.isInitialized = true;
    } catch (error) {
      console.error('Failed to initialize A2AService:', error);
      throw error;
    }
  }

  private initializeDB(): Promise<void> {
    return new Promise((resolve, reject) => {
      const request = indexedDB.open('a2a-db', 1);

      request.onupgradeneeded = (event) => {
        const db = (event.target as IDBOpenDBRequest).result;
        
        // Create object stores if they don't exist
        if (!db.objectStoreNames.contains('agents')) {
          db.createObjectStore('agents', { keyPath: 'id' });
        }
        if (!db.objectStoreNames.contains('tasks')) {
          db.createObjectStore('tasks', { keyPath: 'id' });
        }
        if (!db.objectStoreNames.contains('messages')) {
          db.createObjectStore('messages', { keyPath: 'id' });
        }
      };

      request.onsuccess = (event) => {
        this.db = (event.target as IDBOpenDBRequest).result;
        resolve();
      };

      request.onerror = (event) => {
        const error = (event.target as IDBOpenDBRequest).error;
        console.error('Error opening IndexedDB:', error);
        reject(error);
      };
    });
  }

  private initializeWorker() {
    if (typeof Worker !== 'undefined') {
      this.worker = new Worker(new URL('./a2a-worker.ts', import.meta.url));
      this.worker.onmessage = this.handleWorkerMessage.bind(this);
    }
  }

  private handleWorkerMessage(event: MessageEvent) {
    const { type, payload } = event.data;
    const handlers = this.messageHandlers.get(type);
    if (handlers) {
      handlers.forEach(handler => handler(payload));
    }
  }

  private addMessageHandler(type: string, handler: (data: any) => void) {
    if (!this.messageHandlers.has(type)) {
      this.messageHandlers.set(type, []);
    }
    this.messageHandlers.get(type)?.push(handler);
  }

  private async ensureInitialized(): Promise<void> {
    try {
      await this.initializationPromise;
    } catch (error) {
      throw new Error('Service not initialized: ' + (error as Error).message);
    }
  }

  private async dbOperation<T>(
    storeName: string,
    mode: IDBTransactionMode,
    operation: (store: IDBObjectStore) => IDBRequest<T>
  ): Promise<T> {
    await this.ensureInitialized();

    return new Promise((resolve, reject) => {
      if (!this.db) {
        reject(new Error('Database not initialized'));
        return;
      }

      const transaction = this.db.transaction(storeName, mode);
      const store = transaction.objectStore(storeName);
      const request = operation(store);

      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
  }

  // Agent Management
  async getAgents(): Promise<Agent[]> {
    await this.ensureInitialized();
    return this.dbOperation('agents', 'readonly', store => store.getAll());
  }

  async getAgent(id: string): Promise<Agent> {
    await this.ensureInitialized();
    return this.dbOperation('agents', 'readonly', store => store.get(id));
  }

  async updateAgent(agent: Agent): Promise<void> {
    await this.ensureInitialized();
    await this.dbOperation('agents', 'readwrite', store => store.put(agent));
  }

  // Task Management
  async getTasks(): Promise<Task[]> {
    await this.ensureInitialized();
    return this.dbOperation('tasks', 'readonly', store => store.getAll());
  }

  async getTask(id: string): Promise<Task> {
    await this.ensureInitialized();
    return this.dbOperation('tasks', 'readonly', store => store.get(id));
  }

  async createTask(task: Task): Promise<void> {
    await this.ensureInitialized();
    await this.dbOperation('tasks', 'readwrite', store => store.add(task));
  }

  async updateTask(task: Task): Promise<void> {
    await this.ensureInitialized();
    await this.dbOperation('tasks', 'readwrite', store => store.put(task));
  }

  // Message Management
  async getMessages(agentId: string): Promise<Message[]> {
    await this.ensureInitialized();
    const allMessages = await this.dbOperation<Message[]>('messages', 'readonly', store => store.getAll());
    return allMessages.filter(msg => msg.to === agentId || msg.from === agentId);
  }

  async sendMessage(message: Message): Promise<void> {
    await this.ensureInitialized();
    await this.dbOperation('messages', 'readwrite', store => store.add(message));
  }

  // Agent Communication Implementation
  async receiveMessage(agentId: string): Promise<Message[]> {
    return this.getMessages(agentId);
  }

  async broadcast(message: Message): Promise<void> {
    const agents = await this.getAgents();
    for (const agent of agents) {
      if (agent.id !== message.from) {
        await this.sendMessage({ ...message, to: agent.id });
      }
    }
  }

  // Task Management Implementation
  async assignTask(task: Task): Promise<void> {
    const agents = await this.getAgents();
    const suitableAgent = agents.find(agent => 
      agent.state.status === 'idle' && 
      agent.capabilities.includes(task.type)
    );

    if (suitableAgent) {
      const updatedTask = { ...task, assignedTo: suitableAgent.id, status: 'in_progress' as const };
      await this.updateTask(updatedTask);
      
      // Update agent state
      const updatedAgent = {
        ...suitableAgent,
        state: { ...suitableAgent.state, status: 'busy' as const, currentTask: task.id }
      };
      await this.updateAgent(updatedAgent);

      // Notify agent
      await this.sendMessage({
        id: `task-${task.id}`,
        from: 'system',
        to: suitableAgent.id,
        type: 'notification',
        content: `New task assigned: ${task.type}`,
        timestamp: Date.now(),
        metadata: { task }
      });
    }
  }

  async completeTask(taskId: string, result: any): Promise<void> {
    const task = await this.getTask(taskId);
    if (task && task.assignedTo) {
      const agent = await this.getAgent(task.assignedTo);
      if (agent) {
        // Update task
        await this.updateTask({
          ...task,
          status: 'completed',
          data: { ...task.data, result }
        });

        // Update agent state
        await this.updateAgent({
          ...agent,
          state: { ...agent.state, status: 'idle' as const, currentTask: undefined }
        });

        // Broadcast completion
        await this.broadcast({
          id: `complete-${taskId}`,
          from: agent.id,
          to: 'all',
          type: 'notification',
          content: `Task ${taskId} completed`,
          timestamp: Date.now(),
          metadata: { taskId, result }
        });
      }
    }
  }

  async getTaskStatus(taskId: string): Promise<Task> {
    const task = await this.getTask(taskId);
    if (!task) throw new Error(`Task ${taskId} not found`);
    return task;
  }

  async getAgentTasks(agentId: string): Promise<Task[]> {
    const tasks = await this.getTasks();
    return tasks.filter(task => task.assignedTo === agentId);
  }

  // Alpha Mining Operations
  async generateAlpha(parameters: any): Promise<string> {
    return new Promise((resolve, reject) => {
      this.addMessageHandler('ALPHA_GENERATED', ({ expression }) => resolve(expression));
      this.addMessageHandler('ERROR', ({ error }) => reject(error));
      this.worker?.postMessage({ type: 'GENERATE_ALPHA', payload: parameters });
    });
  }

  async validateAlpha(expression: string): Promise<boolean> {
    return new Promise((resolve, reject) => {
      this.addMessageHandler('ALPHA_VALIDATED', ({ validation }) => resolve(validation === 'valid'));
      this.addMessageHandler('ERROR', ({ error }) => reject(error));
      this.worker?.postMessage({ type: 'VALIDATE_ALPHA', payload: { expression } });
    });
  }

  async optimizeAlpha(expression: string, goals: string[]): Promise<string> {
    return new Promise((resolve, reject) => {
      this.addMessageHandler('ALPHA_OPTIMIZED', ({ optimizedExpression }) => resolve(optimizedExpression));
      this.addMessageHandler('ERROR', ({ error }) => reject(error));
      this.worker?.postMessage({ type: 'OPTIMIZE_ALPHA', payload: { expression, goals } });
    });
  }

  async backtestAlpha(expression: string): Promise<any> {
    return new Promise((resolve, reject) => {
      this.addMessageHandler('ALPHA_BACKTESTED', ({ metrics }) => resolve(metrics));
      this.addMessageHandler('ERROR', ({ error }) => reject(error));
      this.worker?.postMessage({ type: 'BACKTEST_ALPHA', payload: { expression } });
    });
  }

  async constructPortfolio(alphas: string[]): Promise<any> {
    return new Promise((resolve, reject) => {
      this.addMessageHandler('PORTFOLIO_CONSTRUCTED', ({ portfolio }) => resolve(portfolio));
      this.addMessageHandler('ERROR', ({ error }) => reject(error));
      this.worker?.postMessage({ type: 'CONSTRUCT_PORTFOLIO', payload: { alphas } });
    });
  }
} 