import { NextResponse } from 'next/server';
import { openDatabase, putInStore } from '@/lib/indexedDB';

const tools = [
  {
    id: 'ssrn_crawler',
    name: 'ssrn_crawler',
    description: 'Crawls SSRN website to extract article links',
    code: `
      async function crawl(url) {
        const response = await fetch(url);
        const html = await response.text();
        
        const parser = new DOMParser();
        const doc = parser.parseFromString(html, 'text/html');
        
        const articles = Array.from(doc.querySelectorAll('a[href*="/abstract="]')).map(a => ({
          title: a.textContent?.trim(),
          url: a.getAttribute('href')
        }));

        return articles;
      }
    `,
    created_at: Date.now(),
    updated_at: Date.now()
  },
  {
    id: 'pdf_processor',
    name: 'pdf_processor',
    description: 'Processes PDF files to extract text content',
    code: `
      async function process(pdfUrl) {
        const response = await fetch(pdfUrl);
        const pdfBlob = await response.blob();
        
        const pdfText = await new Promise((resolve) => {
          const reader = new FileReader();
          reader.onload = () => resolve(reader.result);
          reader.readAsText(pdfBlob);
        });

        return pdfText;
      }
    `,
    created_at: Date.now(),
    updated_at: Date.now()
  },
  {
    id: 'alpha_idea_generator',
    name: 'alpha_idea_generator',
    description: 'Generates alpha ideas from text content',
    code: `
      function generate(text) {
        const ideas = text.match(/[A-Za-z0-9_]+(?:\s*[+\-*/]\s*[A-Za-z0-9_]+)+/g) || [];
        
        return ideas.map(idea => ({
          expression: idea,
          confidence: Math.random()
        }));
      }
    `,
    created_at: Date.now(),
    updated_at: Date.now()
  },
  {
    id: 'simulation_tool',
    name: 'simulation_tool',
    description: 'Runs simulations based on provided parameters',
    code: `
      async function runSimulation(params) {
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
        } catch (error) {
          throw new Error(\`Simulation failed: \${error.message}\`);
        }
      }

      async function checkProgress(simulationId) {
        try {
          const response = await fetch(\`/api/simulation/progress/\${simulationId}\`);
          
          if (!response.ok) {
            throw new Error('Progress check failed');
          }

          const data = await response.json();
          return {
            status: data.status,
            progress: data.progress || 0,
            result: data.result
          };
        } catch (error) {
          throw new Error(\`Progress check failed: \${error.message}\`);
        }
      }
    `,
    created_at: Date.now(),
    updated_at: Date.now()
  }
];

const contexts = [
  {
    id: 'default_context',
    name: 'default_context',
    data: {
      last_ssrn_crawl: null,
      processed_pdfs: [],
      generated_ideas: [],
      active_simulations: []
    },
    created_at: Date.now(),
    updated_at: Date.now()
  },
  {
    id: 'simulation_context',
    name: 'simulation_context',
    data: {
      default_parameters: {
        universe: 'SP500',
        start_date: '2020-01-01',
        end_date: '2023-12-31',
        lookback_period: 252,
        holding_period: 5,
        transaction_cost: 0.001
      },
      recent_simulations: [],
      performance_metrics: {}
    },
    created_at: Date.now(),
    updated_at: Date.now()
  }
];

export async function POST() {
  try {
    const db = await openDatabase();
    
    // Initialize tools
    for (const tool of tools) {
      await putInStore(db, 'tools', tool);
    }
    
    // Initialize contexts
    for (const context of contexts) {
      await putInStore(db, 'contexts', context);
    }
    
    return NextResponse.json({ 
      success: true,
      message: 'Tools and contexts initialized successfully',
      toolsCount: tools.length,
      contextsCount: contexts.length
    });
  } catch (error) {
    console.error('Error during initialization:', error);
    return NextResponse.json(
      { error: 'Failed to initialize tools and contexts' },
      { status: 500 }
    );
  }
} 