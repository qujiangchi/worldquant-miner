import { NextRequest, NextResponse } from 'next/server';
import { Pinecone } from '@pinecone-database/pinecone';
import { 
  generateRandomVector, 
  formatMetadata, 
  validateVectorData, 
  validateMetadata,
  chunkArray,
  retryWithBackoff
} from '@/lib/pinecone';

// Initialize the Pinecone client
const pc = new Pinecone({
  apiKey: process.env.PINECONE_API_KEY || '',
});

// Get the index
const index = pc.index(process.env.PINECONE_INDEX_NAME || 'worldquant-miner');

// Get the index dimension
let indexDimension = 1024; // Default dimension

// Function to get the index dimension
async function getIndexDimension() {
  try {
    const indexDescription = await pc.describeIndex(process.env.PINECONE_INDEX_NAME || 'worldquant-miner');
    if (indexDescription.dimension) {
      indexDimension = indexDescription.dimension;
      console.log(`Pinecone index dimension: ${indexDimension}`);
    }
    return indexDimension;
  } catch (error) {
    console.error('Error getting index dimension:', error);
    return indexDimension; // Return default if there's an error
  }
}

// Initialize the index dimension
getIndexDimension();

// POST handler for Pinecone operations
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { operation, namespace, data } = body;

    if (!operation) {
      return NextResponse.json({ success: false, error: 'Operation is required' }, { status: 400 });
    }

    switch (operation) {
      case 'upsert': {
        if (!namespace) {
          return NextResponse.json({ success: false, error: 'Namespace is required for upsert operation' }, { status: 400 });
        }

        if (!data || !Array.isArray(data)) {
          return NextResponse.json({ success: false, error: 'Data must be an array for upsert operation' }, { status: 400 });
        }

        // Validate vectors
        for (const vector of data) {
          if (!vector.id) {
            return NextResponse.json({ success: false, error: 'Each vector must have an id' }, { status: 400 });
          }
          if (!vector.values || !Array.isArray(vector.values)) {
            return NextResponse.json({ success: false, error: 'Each vector must have a values array' }, { status: 400 });
          }
          if (vector.values.length !== indexDimension) {
            return NextResponse.json({ 
              success: false, 
              error: `Vector dimension mismatch. Expected ${indexDimension}, got ${vector.values.length}` 
            }, { status: 400 });
          }
        }

        // Upsert vectors to Pinecone
        const result = await retryWithBackoff(() => index.namespace(namespace).upsert(data));
        
        return NextResponse.json({ success: true, result });
      }

      case 'query': {
        if (!namespace) {
          return NextResponse.json({ success: false, error: 'Namespace is required for query operation' }, { status: 400 });
        }

        if (!data || !data.vector) {
          return NextResponse.json({ success: false, error: 'Vector is required for query operation' }, { status: 400 });
        }

        const { vector, topK = 5, filter } = data;

        // Validate vector
        if (!validateVectorData(vector)) {
          return NextResponse.json({ success: false, error: 'Invalid vector data' }, { status: 400 });
        }

        // Adjust the query vector dimension if needed
        let adjustedVector = vector;
        if (vector.length !== indexDimension) {
          console.log(`Adjusting query vector dimension from ${vector.length} to ${indexDimension}`);
          if (vector.length > indexDimension) {
            adjustedVector = vector.slice(0, indexDimension);
          } else {
            adjustedVector = [...vector, ...generateRandomVector(indexDimension - vector.length)];
          }
        }

        const response = await retryWithBackoff(() => 
          index.namespace(namespace).query({
            topK,
            vector: adjustedVector,
            includeValues: true,
            includeMetadata: true,
            filter,
          })
        );

        return NextResponse.json({ success: true, response });
      }

      case 'delete': {
        if (!namespace) {
          return NextResponse.json({ success: false, error: 'Namespace is required for delete operation' }, { status: 400 });
        }

        if (!data || !Array.isArray(data.ids)) {
          return NextResponse.json({ success: false, error: 'IDs array is required for delete operation' }, { status: 400 });
        }

        const { ids } = data;

        // Delete vectors in chunks to avoid rate limits
        const chunks = chunkArray(ids, 100);
        const results = [];

        for (const chunk of chunks) {
          const result = await retryWithBackoff(() => index.namespace(namespace).deleteMany(chunk));
          results.push(result);
        }

        return NextResponse.json({ success: true, results });
      }

      case 'deleteAll': {
        if (!namespace) {
          return NextResponse.json({ success: false, error: 'Namespace is required for deleteAll operation' }, { status: 400 });
        }

        const result = await retryWithBackoff(() => index.namespace(namespace).deleteAll());
        return NextResponse.json({ success: true, result });
      }

      default:
        return NextResponse.json({ success: false, error: 'Invalid operation' }, { status: 400 });
    }
  } catch (error) {
    console.error('Error processing Pinecone operation:', error);
    return NextResponse.json({ 
      success: false, 
      error: error instanceof Error ? error.message : 'Internal server error' 
    }, { status: 500 });
  }
} 