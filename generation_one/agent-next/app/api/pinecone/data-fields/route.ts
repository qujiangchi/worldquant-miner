import { NextRequest, NextResponse } from 'next/server';
import { pc, index, EmbeddingsResponse, DenseEmbedding } from '@/lib/pinecone';
import { retryWithBackoff } from '@/lib/pinecone';

// Base data field data
const baseDataFields = [
  {
    id: 'df1',
    name: 'Price',
    category: 'Market',
    description: 'Stock price data',
    status: 'Pending',
    lastUploaded: null,
    vectorCount: 0,
    namespace: 'data-fields'
  },
  {
    id: 'df2',
    name: 'Volume',
    category: 'Market',
    description: 'Trading volume data',
    status: 'Pending',
    lastUploaded: null,
    vectorCount: 0,
    namespace: 'data-fields'
  },
  {
    id: 'df3',
    name: 'Market Cap',
    category: 'Fundamental',
    description: 'Company market capitalization',
    status: 'Pending',
    lastUploaded: null,
    vectorCount: 0,
    namespace: 'data-fields'
  }
];

// GET handler to list all data fields
export async function GET() {
  try {
    // Get vector count for each data field from Pinecone
    const updatedDataFields = await Promise.all(
      baseDataFields.map(async (dataField) => {
        try {
          // Query the namespace to get vector count
          const stats = await retryWithBackoff(() => index.namespace(dataField.namespace).describeIndexStats());
          const vectorCount = stats.namespaces?.[dataField.namespace]?.recordCount || 0;
          
          // Determine status based on vector count
          let status: 'Pending' | 'In Progress' | 'Uploaded' | 'Error' = 'Pending';
          if (vectorCount > 0) {
            status = 'Uploaded';
          }
          
          return {
            ...dataField,
            status,
            vectorCount,
            lastUploaded: vectorCount > 0 ? new Date().toISOString() : null
          };
        } catch (error) {
          console.error(`Error getting vector count for data field ${dataField.id}:`, error);
          return dataField;
        }
      })
    );
    
    return NextResponse.json({ success: true, dataFields: updatedDataFields });
  } catch (error) {
    console.error('Error getting data fields:', error);
    return NextResponse.json({ success: false, error: 'Internal server error' }, { status: 500 });
  }
}

// POST handler to upload data field data
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { dataFieldId, dataFieldInfo } = body;

    if (!dataFieldId) {
      return NextResponse.json({ success: false, error: 'Data field ID is required' }, { status: 400 });
    }

    if (!dataFieldInfo) {
      return NextResponse.json({ success: false, error: 'Data field information is required' }, { status: 400 });
    }

    const { name, category, description } = dataFieldInfo;

    if (!name || !category) {
      return NextResponse.json({ success: false, error: 'Data field name and category are required' }, { status: 400 });
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
    ) as unknown as EmbeddingsResponse;

    // Create vectors with the generated embeddings
    const vectors = [{
      id: dataFieldId,
      values: (embeddings.data[0] as any).values,
      metadata: {
        name: dataFieldInfo.name,
        category: dataFieldInfo.category,
        description: dataFieldInfo.description,
        timestamp: new Date().toISOString()
      }
    }];

    // Upload the vectors to Pinecone
    const result = await retryWithBackoff(
      () => index.namespace('data-fields').upsert(vectors),
      3, // maxRetries
      1000 // initialDelay
    );
    
    return NextResponse.json({ success: true, result });
  } catch (error) {
    console.error('Error uploading data field to Pinecone:', error);
    return NextResponse.json({ 
      success: false, 
      error: error instanceof Error ? error.message : 'Internal server error' 
    }, { status: 500 });
  }
} 