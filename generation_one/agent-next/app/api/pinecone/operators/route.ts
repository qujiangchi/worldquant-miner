import { NextRequest, NextResponse } from 'next/server';
import { retryWithBackoff, DenseEmbedding } from '@/lib/pinecone';
import { Pinecone } from '@pinecone-database/pinecone';

// Initialize the Pinecone client
const pc = new Pinecone({
  apiKey: process.env.PINECONE_API_KEY || '',
});

// Get the index
const index = pc.index(process.env.PINECONE_INDEX_NAME || 'worldquant-miner');

// Base operator data
const baseOperators = [
  {
    id: 'op1',
    name: 'Momentum Alpha',
    category: 'Momentum',
    description: 'Alpha factor based on price momentum',
    status: 'Pending',
    lastUploaded: null,
    vectorCount: 0,
    namespace: 'operators'
  },
  {
    id: 'op2',
    name: 'Mean Reversion',
    category: 'Statistical',
    description: 'Alpha factor based on mean reversion',
    status: 'Pending',
    lastUploaded: null,
    vectorCount: 0,
    namespace: 'operators'
  },
  {
    id: 'op3',
    name: 'Volume Profile',
    category: 'Volume',
    description: 'Alpha factor based on volume profile',
    status: 'Pending',
    lastUploaded: null,
    vectorCount: 0,
    namespace: 'operators'
  }
];

// GET handler to list all operators
export async function GET() {
  try {
    // Get vector count for each operator from Pinecone
    const updatedOperators = await Promise.all(
      baseOperators.map(async (operator) => {
        try {
          // Query the namespace to get vector count
          const stats = await retryWithBackoff(() => index.namespace(operator.namespace).describeIndexStats());
          const vectorCount = stats.namespaces?.[operator.namespace]?.recordCount || 0;
          
          // Determine status based on vector count
          let status: 'Pending' | 'In Progress' | 'Uploaded' | 'Error' = 'Pending';
          if (vectorCount > 0) {
            status = 'Uploaded';
          }
          
          return {
            ...operator,
            status,
            vectorCount,
            lastUploaded: vectorCount > 0 ? new Date().toISOString() : null
          };
        } catch (error) {
          console.error(`Error getting vector count for operator ${operator.id}:`, error);
          return operator;
        }
      })
    );
    
    return NextResponse.json({ success: true, operators: updatedOperators });
  } catch (error) {
    console.error('Error getting operators:', error);
    return NextResponse.json({ success: false, error: 'Internal server error' }, { status: 500 });
  }
}

// POST handler to upload operator data
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { operatorId, operatorInfo } = body;

    if (!operatorId) {
      return NextResponse.json({ success: false, error: 'Operator ID is required' }, { status: 400 });
    }

    if (!operatorInfo) {
      return NextResponse.json({ success: false, error: 'Operator information is required' }, { status: 400 });
    }

    const { name, category, description } = operatorInfo;

    if (!name || !category) {
      return NextResponse.json({ success: false, error: 'Operator name and category are required' }, { status: 400 });
    }

    // Generate embeddings using Pinecone's inference API
    const textToEmbed = `${name} ${category} ${description || ''}`;
    const embeddings = await pc.inference.embed(
      "multilingual-e5-large",
      [textToEmbed],
      {
        input_type: "passage",
        truncate: "END"
      }
    );

    // Create vectors with the generated embeddings
    const vectors = [{
      id: operatorId,
      values: (embeddings.data[0] as any).values,
      metadata: {
        name: operatorInfo.name,
        category: operatorInfo.category,
        description: operatorInfo.description,
        timestamp: new Date().toISOString()
      }
    }];

    // Upload the vectors to Pinecone
    const result = await retryWithBackoff(() => index.namespace('operators').upsert(vectors));
    
    return NextResponse.json({ success: true, result });
  } catch (error) {
    console.error('Error uploading operator to Pinecone:', error);
    return NextResponse.json({ 
      success: false, 
      error: error instanceof Error ? error.message : 'Internal server error' 
    }, { status: 500 });
  }
} 