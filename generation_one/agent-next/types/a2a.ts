export interface Agent {
  id: string;
  name: string;
  role: string;
  capabilities: string[];
  state: AgentState;
}

export interface AgentState {
  status: 'idle' | 'busy' | 'error';
  currentTask?: string;
  lastActivity: number;
}

export interface Task {
  id: string;
  type: 'alpha_generation' | 'alpha_validation' | 'alpha_optimization' | 'portfolio_construction';
  status: 'pending' | 'in_progress' | 'completed' | 'failed';
  priority: 'low' | 'medium' | 'high';
  assignedTo?: string;
  createdAt: number;
  updatedAt: number;
  data: any;
}

export interface Message {
  id: string;
  from: string;
  to: string;
  type: 'request' | 'response' | 'notification' | 'error';
  content: string;
  timestamp: number;
  metadata?: any;
}

export interface AlphaMiningTask extends Task {
  data: {
    expression: string;
    parameters: Record<string, any>;
    constraints: {
      sharpe?: number;
      turnover?: number;
      fitness?: number;
    };
    optimizationGoals: string[];
  };
}

export interface AgentCapabilities {
  canGenerateAlphas: boolean;
  canValidateAlphas: boolean;
  canOptimizeAlphas: boolean;
  canConstructPortfolios: boolean;
  canAnalyzeRisk: boolean;
  canBacktest: boolean;
}

export interface AgentCommunication {
  sendMessage(message: Message): Promise<void>;
  receiveMessage(agentId: string): Promise<Message[]>;
  broadcast(message: Message): Promise<void>;
}

export interface AgentTaskManagement {
  assignTask(task: Task): Promise<void>;
  completeTask(taskId: string, result: any): Promise<void>;
  getTaskStatus(taskId: string): Promise<Task>;
  getAgentTasks(agentId: string): Promise<Task[]>;
}

export interface AlphaMiningProtocol {
  generateAlpha(parameters: any): Promise<string>;
  validateAlpha(expression: string): Promise<boolean>;
  optimizeAlpha(expression: string, goals: string[]): Promise<string>;
  backtestAlpha(expression: string): Promise<any>;
  constructPortfolio(alphas: string[]): Promise<any>;
} 