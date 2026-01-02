// Client-side utility to interact with the Pinecone API

/**
 * Upsert vectors to a namespace in Pinecone
 * @param namespace The namespace to upsert vectors to
 * @param vectors The vectors to upsert
 * @returns The result of the upsert operation
 */
export async function upsertVectors(namespace: string, vectors: any[]) {
  try {
    // Validate input
    if (!namespace) {
      throw new Error('Namespace is required');
    }
    if (!vectors || !Array.isArray(vectors)) {
      throw new Error('Vectors must be an array');
    }

    // Transform vectors to match Pinecone format
    const pineconeVectors = vectors.map(vector => {
      // Ensure each vector has required fields
      if (!vector.id) {
        throw new Error('Each vector must have an id');
      }
      if (!vector.values || !Array.isArray(vector.values)) {
        throw new Error('Each vector must have a values array');
      }

      return {
        id: vector.id,
        values: vector.values,
        metadata: vector.metadata || {},
        sparseValues: vector.sparseValues || undefined
      };
    });

    // Process in batches of 100 to avoid rate limits
    const batchSize = 100;
    const results = [];
    
    for (let i = 0; i < pineconeVectors.length; i += batchSize) {
      const batch = pineconeVectors.slice(i, i + batchSize);
      
      const response = await fetch('/api/pinecone', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          operation: 'upsert',
          namespace,
          data: batch,
        }),
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to upsert vectors');
      }
      
      const result = await response.json();
      results.push(result);
    }
    
    return { success: true, results };
  } catch (error) {
    console.error('Error upserting vectors:', error);
    throw error;
  }
}

/**
 * Query vectors from a namespace in Pinecone
 * @param namespace The namespace to query vectors from
 * @param vector The vector to query
 * @param topK The number of results to return
 * @param filter Optional filter to apply to the query
 * @returns The result of the query operation
 */
export async function queryVectors(
  namespace: string,
  vector: number[],
  topK: number = 5,
  filter?: any
) {
  try {
    const response = await fetch('/api/pinecone', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        operation: 'query',
        namespace,
        data: {
          vector,
          topK,
          filter,
        },
      }),
    });
    
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || 'Failed to query vectors');
    }
    
    return await response.json();
  } catch (error) {
    console.error('Error querying vectors:', error);
    throw error;
  }
}

/**
 * Delete vectors from a namespace in Pinecone
 * @param namespace The namespace to delete vectors from
 * @param ids The IDs of the vectors to delete
 * @returns The result of the delete operation
 */
export async function deleteVectors(namespace: string, ids: string[]) {
  try {
    const response = await fetch('/api/pinecone', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        operation: 'delete',
        namespace,
        data: ids,
      }),
    });
    
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || 'Failed to delete vectors');
    }
    
    return await response.json();
  } catch (error) {
    console.error('Error deleting vectors:', error);
    throw error;
  }
}

/**
 * Delete all vectors from a namespace in Pinecone
 * @param namespace The namespace to delete all vectors from
 * @returns The result of the deleteAll operation
 */
export async function deleteAllVectors(namespace: string) {
  try {
    const response = await fetch('/api/pinecone', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        operation: 'deleteAll',
        namespace,
      }),
    });
    
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || 'Failed to delete all vectors');
    }
    
    return await response.json();
  } catch (error) {
    console.error('Error deleting all vectors:', error);
    throw error;
  }
}

export default {
  upsertVectors,
  queryVectors,
  deleteVectors,
  deleteAllVectors,
}; 