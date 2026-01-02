import { Agent, Task, Message } from '../types/a2a';
import { createChatCompletion } from './deepseek';

// Initialize IndexedDB
const request = indexedDB.open('a2a-db', 1);

request.onupgradeneeded = (event) => {
  const db = (event.target as IDBOpenDBRequest).result;
  
  // Create object stores
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

// Handle messages from the main thread
self.onmessage = async (event) => {
  const { type, payload } = event.data;

  switch (type) {
    case 'GENERATE_ALPHA':
      try {
        const response = await createChatCompletion([
          {
            role: 'system',
            content: 'You are an expert in quantitative finance and alpha generation. Generate a new alpha expression based on the given parameters.'
          },
          {
            role: 'user',
            content: JSON.stringify(payload)
          }
        ]);
        
        self.postMessage({
          type: 'ALPHA_GENERATED',
          payload: { expression: response }
        });
      } catch (error: unknown) {
        self.postMessage({
          type: 'ERROR',
          payload: { error: error instanceof Error ? error.message : 'Unknown error' }
        });
      }
      break;

    case 'VALIDATE_ALPHA':
      try {
        const response = await createChatCompletion([
          {
            role: 'system',
            content: 'You are an expert in quantitative finance. Validate the given alpha expression and provide feedback.'
          },
          {
            role: 'user',
            content: payload.expression
          }
        ]);
        
        self.postMessage({
          type: 'ALPHA_VALIDATED',
          payload: { validation: response }
        });
      } catch (error: unknown) {
        self.postMessage({
          type: 'ERROR',
          payload: { error: error instanceof Error ? error.message : 'Unknown error' }
        });
      }
      break;

    case 'OPTIMIZE_ALPHA':
      try {
        const response = await createChatCompletion([
          {
            role: 'system',
            content: 'You are an expert in quantitative finance. Optimize the given alpha expression based on the specified goals.'
          },
          {
            role: 'user',
            content: JSON.stringify(payload)
          }
        ]);
        
        self.postMessage({
          type: 'ALPHA_OPTIMIZED',
          payload: { optimizedExpression: response }
        });
      } catch (error: unknown) {
        self.postMessage({
          type: 'ERROR',
          payload: { error: error instanceof Error ? error.message : 'Unknown error' }
        });
      }
      break;

    case 'BACKTEST_ALPHA':
      try {
        const response = await createChatCompletion([
          {
            role: 'system',
            content: 'You are an expert in quantitative finance. Perform a backtest on the given alpha expression and provide detailed metrics.'
          },
          {
            role: 'user',
            content: payload.expression
          }
        ]);
        
        self.postMessage({
          type: 'ALPHA_BACKTESTED',
          payload: { metrics: JSON.parse(response) }
        });
      } catch (error: unknown) {
        self.postMessage({
          type: 'ERROR',
          payload: { error: error instanceof Error ? error.message : 'Unknown error' }
        });
      }
      break;

    case 'CONSTRUCT_PORTFOLIO':
      try {
        const response = await createChatCompletion([
          {
            role: 'system',
            content: 'You are an expert in portfolio construction. Create an optimal portfolio using the given alpha expressions.'
          },
          {
            role: 'user',
            content: JSON.stringify(payload)
          }
        ]);
        
        self.postMessage({
          type: 'PORTFOLIO_CONSTRUCTED',
          payload: { portfolio: JSON.parse(response) }
        });
      } catch (error: unknown) {
        self.postMessage({
          type: 'ERROR',
          payload: { error: error instanceof Error ? error.message : 'Unknown error' }
        });
      }
      break;
  }
}; 