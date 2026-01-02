import { NextRequest, NextResponse } from 'next/server';
import { Pinecone } from '@pinecone-database/pinecone';
import { retryWithBackoff } from '@/lib/pinecone';

// Initialize the Pinecone client
const pc = new Pinecone({
  apiKey: process.env.PINECONE_API_KEY || '',
});

// GET handler to list all indexes
export async function GET() {
  try {
    // Get all indexes from Pinecone
    const response = await retryWithBackoff(() => pc.listIndexes());
    
    console.log('Raw indexes from Pinecone:', response);

    // Extract the indexes array from the response
    const indexes = response.indexes || [];
    console.log('Extracted indexes array:', indexes);

    
    return NextResponse.json({ success: true, indexes: indexes });
  } catch (error) {
    console.error('Error getting Pinecone indexes:', error);
    return NextResponse.json({ success: false, error: 'Internal server error' }, { status: 500 });
  }
}

// POST handler for index operations
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { operation, indexName, config } = body;

    if (!operation) {
      return NextResponse.json({ success: false, error: 'Operation is required' }, { status: 400 });
    }

    if (!indexName) {
      return NextResponse.json({ success: false, error: 'Index name is required' }, { status: 400 });
    }

    switch (operation) {
      case 'describe': {
        // Get the index description from Pinecone
        const indexDescription = await retryWithBackoff(() => pc.describeIndex(indexName));
        
        // Get vector count for the index
        const index = pc.index(indexName);
        const stats = await retryWithBackoff(() => index.describeIndexStats());
        const vectorCount = stats.totalRecordCount || 0;
        
        // Map the Pinecone index to our PineconeIndex interface
        const indexData = {
          id: indexName,
          dimension: indexDescription.dimension || 0,
          metric: indexDescription.metric || 'cosine',
          pods: 1, // Default value as this might not be available in the API
          replicas: 1, // Default value as this might not be available in the API
          podType: 'p1.x1', // Default value as this might not be available in the API
          status: indexDescription.status?.state || 'Unknown',
          vectorCount,
          lastUpdated: new Date().toISOString(),
          deletion_protection: indexDescription.deletionProtection ? 'enabled' : 'disabled',
          tags: indexDescription.tags || {},
          vector_type: 'dense', // Default value
          host: indexDescription.host || '',
          spec: {
            serverless: indexDescription.spec?.serverless ? {
              cloud: indexDescription.spec.serverless.cloud || '',
              region: indexDescription.spec.serverless.region || ''
            } : undefined
          }
        };
        
        return NextResponse.json({ success: true, index: indexData });
      }

      case 'delete': {
        // Check if deletion protection is enabled
        const indexDescription = await retryWithBackoff(() => pc.describeIndex(indexName));
        if (indexDescription.deletionProtection) {
          return NextResponse.json({ 
            success: false, 
            error: { message: 'Deletion protection is enabled for this index. Disable deletion protection before retrying.' } 
          }, { status: 400 });
        }
        
        // Delete the index
        await retryWithBackoff(() => pc.deleteIndex(indexName));
        return NextResponse.json({ success: true });
      }

      case 'configure': {
        if (!config) {
          return NextResponse.json({ success: false, error: 'Configuration is required for configure operation' }, { status: 400 });
        }

        // Configure the index
        if (config.deletion_protection !== undefined) {
          await retryWithBackoff(() => pc.configureIndex(indexName, {
            deletionProtection: config.deletion_protection === 'enabled' ? true : false
          } as any));
        }
        
        if (config.tags !== undefined) {
          await retryWithBackoff(() => pc.configureIndex(indexName, {
            tags: config.tags
          }));
        }
        
        return NextResponse.json({ success: true });
      }

      case 'create': {
        if (!config) {
          return NextResponse.json({ success: false, error: 'Configuration is required for create operation' }, { status: 400 });
        }

        const { dimension, metric, deletion_protection, tags } = config;

        if (!dimension) {
          return NextResponse.json({ success: false, error: 'Dimension is required for create operation' }, { status: 400 });
        }

        // Check if the index already exists
        const indexes = await retryWithBackoff(() => pc.listIndexes());
        if (Array.from(indexes as any).some((idx: any) => idx.name === indexName)) {
          return NextResponse.json({ success: false, error: { message: 'Index already exists' } }, { status: 400 });
        }
        
        // Create the index
        await retryWithBackoff(() => pc.createIndex({
          name: indexName,
          dimension,
          metric: metric || 'cosine',
          deletionProtection: deletion_protection === 'enabled' ? true : false,
          tags: tags || {}
        } as any));
        
        return NextResponse.json({ success: true });
      }

      default:
        return NextResponse.json({ success: false, error: `Unsupported operation: ${operation}` }, { status: 400 });
    }
  } catch (error) {
    console.error('Error processing Pinecone index operation:', error);
    return NextResponse.json({ success: false, error: 'Internal server error' }, { status: 500 });
  }
} 