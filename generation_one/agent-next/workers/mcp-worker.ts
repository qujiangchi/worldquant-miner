// Web worker for MCP agent
let db: IDBDatabase | null = null;

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

// Initialize IndexedDB
const initDB = () => {
  return new Promise<IDBDatabase>((resolve, reject) => {
    const request = indexedDB.open('mcp-db', 1);
    
    request.onerror = () => reject(request.error);
    request.onsuccess = () => {
      db = request.result;
      resolve(db);
    };
    
    request.onupgradeneeded = (event) => {
      const db = (event.target as IDBOpenDBRequest).result;
      
      if (!db.objectStoreNames.contains('tools')) {
        const toolsStore = db.createObjectStore('tools', { keyPath: 'id' });
        toolsStore.createIndex('name', 'name', { unique: true });
      }
      
      if (!db.objectStoreNames.contains('contexts')) {
        const contextsStore = db.createObjectStore('contexts', { keyPath: 'id' });
        contextsStore.createIndex('name', 'name', { unique: true });
      }

      if (!db.objectStoreNames.contains('tool_chains')) {
        const chainsStore = db.createObjectStore('tool_chains', { keyPath: 'id' });
        chainsStore.createIndex('name', 'name', { unique: true });
      }
    };
  });
};

// Get all items from a store
const getAllFromStore = (storeName: string) => {
  return new Promise<any[]>((resolve, reject) => {
    if (!db) {
      reject(new Error('Database not initialized'));
      return;
    }

    const transaction = db.transaction(storeName, 'readonly');
    const store = transaction.objectStore(storeName);
    const request = store.getAll();

    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
};

// Get a single item from a store
const getFromStore = (storeName: string, key: string) => {
  return new Promise<any>((resolve, reject) => {
    if (!db) {
      reject(new Error('Database not initialized'));
      return;
    }

    const transaction = db.transaction(storeName, 'readonly');
    const store = transaction.objectStore(storeName);
    const request = store.get(key);

    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
};

// Add or update an item in a store
const putInStore = (storeName: string, item: any) => {
  return new Promise<void>((resolve, reject) => {
    if (!db) {
      reject(new Error('Database not initialized'));
      return;
    }

    const transaction = db.transaction(storeName, 'readwrite');
    const store = transaction.objectStore(storeName);
    const request = store.put(item);

    request.onsuccess = () => resolve();
    request.onerror = () => reject(request.error);
  });
};

// SSRN Crawler Tool
async function ssrnCrawler(input: { url: string }): Promise<SSRNResponse> {
  try {
    // Fetch the list of papers from our API endpoint
    const response = await fetch('/api/mcp/ssrn');
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    const papers = data.papers || [];

    // Process each paper using our API endpoint
    const processedPapers = await Promise.all(
      papers.map(async (paper: any) => {
        try {
          // Use our API endpoint for paper details
          const paperResponse = await fetch(`/api/mcp/ssrn?paperId=${paper.id}`);
          
          if (!paperResponse.ok) {
            console.warn(`Failed to fetch details for paper ${paper.id}: ${paperResponse.status}`);
            return {
              id: paper.id,
              title: paper.title,
              authors: paper.authors || [],
              abstract: paper.abstract || '',
              url: `https://papers.ssrn.com/sol3/papers.cfm?abstract_id=${paper.id}`,
              pdfUrl: null,
              publicationStatus: paper.publicationStatus || 'Unknown',
              pageCount: paper.pageCount || 0,
              downloads: paper.downloads || 0,
              approvedDate: paper.approvedDate || ''
            };
          }

          const paperDetails = await paperResponse.json();
          
          return {
            id: paper.id,
            title: paper.title,
            authors: paper.authors || [],
            abstract: paper.abstract || '',
            url: `https://papers.ssrn.com/sol3/papers.cfm?abstract_id=${paper.id}`,
            pdfUrl: paperDetails.pdfUrl || null,
            publicationStatus: paper.publicationStatus || 'Unknown',
            pageCount: paper.pageCount || 0,
            downloads: paper.downloads || 0,
            approvedDate: paper.approvedDate || ''
          };
        } catch (error) {
          console.error(`Error processing paper ${paper.id}:`, error);
          return {
            id: paper.id,
            title: paper.title,
            authors: paper.authors || [],
            abstract: paper.abstract || '',
            url: `https://papers.ssrn.com/sol3/papers.cfm?abstract_id=${paper.id}`,
            pdfUrl: null,
            publicationStatus: paper.publicationStatus || 'Unknown',
            pageCount: paper.pageCount || 0,
            downloads: paper.downloads || 0,
            approvedDate: paper.approvedDate || ''
          };
        }
      })
    );

    return {
      total: data.total || processedPapers.length,
      papers: processedPapers
    };
  } catch (error) {
    console.error('Error in SSRN crawler:', error);
    throw new Error(`Failed to fetch SSRN papers: ${error instanceof Error ? error.message : 'Unknown error'}`);
  }
}

// PDF Processor Tool
const pdfProcessor = async (pdfUrl: string) => {
  try {
    const response = await fetch(pdfUrl);
    const pdfBlob = await response.blob();
    
    // Extract text from PDF
    const pdfText = await new Promise<string>((resolve) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result as string);
      reader.readAsText(pdfBlob);
    });

    return pdfText;
  } catch (error) {
    throw new Error(`Failed to process PDF: ${error}`);
  }
};

// Alpha Idea Generator Tool
const alphaIdeaGenerator = async (text: string) => {
  try {
    // Extract potential alpha ideas from text
    const ideas = text.match(/[A-Za-z0-9_]+(?:\s*[+\-*/]\s*[A-Za-z0-9_]+)+/g) || [];
    
    return ideas.map(idea => ({
      expression: idea,
      confidence: Math.random() // Placeholder for confidence score
    }));
  } catch (error) {
    throw new Error(`Failed to generate alpha ideas: ${error}`);
  }
};

// Simulation Tool
const simulationTool = async (params: any) => {
  try {
    const response = await fetch('/api/simulation/run', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(params)
    });

    if (!response.ok) {
      throw new Error('Simulation request failed');
    }

    const data = await response.json();
    return {
      simulationId: data.simulationId,
      status: data.status,
      progress: data.progress || 0
    };
  } catch (error: unknown) {
    const errorMessage = error instanceof Error ? error.message : 'Unknown error';
    throw new Error(`Simulation failed: ${errorMessage}`);
  }
};

// Function to execute a tool based on its name
async function executeTool(tool: any, input: any) {
  try {
    switch (tool.name) {
      case 'ssrn_crawler':
        if (!input?.url) {
          throw new Error('URL is required for SSRN crawler');
        }
        return await ssrnCrawler(input);
      
      case 'pdf_processor':
        if (!input?.pdfUrl) {
          throw new Error('PDF URL is required for PDF processor');
        }
        return await pdfProcessor(input.pdfUrl);
      
      case 'alpha_idea_generator':
        if (!input?.text) {
          throw new Error('Text is required for alpha idea generator');
        }
        return await alphaIdeaGenerator(input.text);
      
      case 'simulation_tool':
        if (!input?.params) {
          throw new Error('Parameters are required for simulation tool');
        }
        return await simulationTool(input.params);
      
      default:
        throw new Error(`Unknown tool: ${tool.name}`);
    }
  } catch (error) {
    console.error(`Error executing tool ${tool.name}:`, error);
    throw error;
  }
}

// Handle messages from the main thread
self.onmessage = async (event: MessageEvent) => {
  try {
    if (!db) {
      db = await initDB();
    }

    const { type, data: messageData } = event.data;

    switch (type) {
      case 'initialize':
        // Initialize tools and contexts
        for (const tool of messageData.tools) {
          await putInStore('tools', {
            ...tool,
            created_at: Date.now(),
            updated_at: Date.now()
          });
        }

        for (const context of messageData.contexts) {
          await putInStore('contexts', {
            ...context,
            created_at: Date.now(),
            updated_at: Date.now()
          });
        }

        // Send initialization complete message
        self.postMessage({
          type: 'initialization_complete',
          data: {
            tools: messageData.tools.length,
            contexts: messageData.contexts.length
          }
        });
        break;

      case 'execute_tool':
        const tool = await getFromStore('tools', messageData.toolId);
        if (!tool) {
          throw new Error(`Tool not found: ${messageData.toolId}`);
        }

        // Execute the tool with input data
        const result = await executeTool(tool, messageData.input);

        // Send the result back to the main thread
        self.postMessage({
          type: 'tool_result',
          data: {
            toolId: tool.id,
            result
          }
        });
        break;

      case 'update_context':
        await putInStore('contexts', messageData);
        break;

      default:
        throw new Error(`Unknown message type: ${type}`);
    }
  } catch (error) {
    console.error('Error in web worker:', error);
    self.postMessage({
      type: 'error',
      data: {
        message: error instanceof Error ? error.message : 'Unknown error occurred',
        toolId: event.data.data?.toolId
      }
    });
  }
}; 