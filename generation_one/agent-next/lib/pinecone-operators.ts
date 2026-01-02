import { Pinecone } from '@pinecone-database/pinecone';
import { getStoredCredentials } from './auth';

// Types for operator and data field data
export interface Operator {
  id: string;
  name: string;
  category: string;
  description?: string;
  status: 'Pending' | 'In Progress' | 'Uploaded' | 'Error';
  lastUploaded: string | null;
  vectorCount: number;
  namespace: string;
}

export interface DataField {
  id: string;
  name: string;
  category: string;
  description?: string;
  definition?: string;
  status: 'Pending' | 'In Progress' | 'Uploaded' | 'Error';
  lastUploaded: string | null;
  vectorCount: number;
  namespace: string;
}

export interface PineconeIndex {
  id: string;
  dimension: number;
  metric: string;
  pods: number;
  replicas: number;
  podType: string;
  status: string;
  vectorCount: number;
  lastUpdated: string;
  deletion_protection?: 'enabled' | 'disabled';
  tags?: Record<string, string>;
  vector_type?: 'dense' | 'sparse';
  host?: string;
  spec?: {
    serverless?: {
      cloud: string;
      region: string;
    };
  };
}

// Function to get all operators
export async function getOperators(): Promise<Operator[]> {
  try {
    // Get the JWT token
    const jwtToken = localStorage.getItem('worldquant_jwt');
    
    if (!jwtToken) {
      return [];
    }
    
    // POST
    const response = await fetch('/api/operators', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        jwtToken,
      }),
    });
    const data = await response.json();
    console.log('Operators data:', data);
    if (data) {
      return data;
    } else {
      console.error('Error getting operators:', data.error);
      return [];
    }
  } catch (error) {
    console.error('Error getting operators:', error);
    return [];
  }
}

export async function getAllDataFields(): Promise<DataField[]> {
  try {
    const jwtToken = localStorage.getItem('worldquant_jwt');
    
    if (!jwtToken) {
      console.error('No JWT token found');
      return [];
    }
    
    const datasets = ['fundamental6', 'fundamental2', 'analyst4', 'model16', 'model51', 'news12'];
    const allDataFields: DataField[] = [];
    
    // Helper function to add delay between requests
    const delay = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));
    
    // Process each dataset
    for (const dataset of datasets) {
      console.log(`Fetching data fields for dataset: ${dataset}`);
      
      let offset = 0;
      const limit = 20;
      let totalCount = 0;
      let hasMore = true;
      
      // First request to get the total count
      const initialResponse = await fetch('/api/data-fields', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          jwtToken,
          dataset,
          limit: '1',
          instrumentType: 'EQUITY',
          region: 'USA',
          universe: 'TOP3000',
          delay: '1',
          offset: '0',
        }),
      });
      
      if (!initialResponse.ok) {
        console.error(`Failed to fetch initial data for dataset ${dataset}:`, initialResponse.status, initialResponse.statusText);
        continue;
      }
      
      const initialData = await initialResponse.json();
      
      if (initialData && initialData.count) {
        totalCount = initialData.count;
        console.log(`Dataset ${dataset} has ${totalCount} data fields`);
        
        // Add initial results if any
        if (initialData.results && initialData.results.length > 0) {
          allDataFields.push(...initialData.results);
        }
        
        // Add delay before next request
        await delay(600);
        
        // Continue fetching with pagination
        while (hasMore && offset + limit < totalCount) {
          offset += limit;
          
          console.log(`Fetching data fields for dataset ${dataset} (offset: ${offset}, limit: ${limit})`);
          
          const response = await fetch('/api/data-fields', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({
              jwtToken,
              dataset,
              limit: limit.toString(),
              instrumentType: 'EQUITY',
              region: 'USA',
              universe: 'TOP3000',
              delay: '1',
              offset: offset.toString(),
            }),
          });
          
          if (!response.ok) {
            console.error(`Failed to fetch data fields for dataset ${dataset}:`, response.status, response.statusText);
            break;
          }
          
          const data = await response.json();
          
          if (data && data.results && data.results.length > 0) {
            allDataFields.push(...data.results);
            console.log(`Added ${data.results.length} data fields from dataset ${dataset}`);
          } else {
            hasMore = false;
          }
          
          // Add delay before next request
          await delay(600);
        }
      } else {
        console.error(`No count information for dataset ${dataset}`);
      }
    }
    
    console.log(`Total data fields fetched: ${allDataFields.length}`);
    
    // Map the API response to our DataField interface
    return allDataFields.map((field: any) => ({
      id: field.id || '',
      name: field.description || field.id || '',
      category: field.category?.name || field.category || '',
      description: field.description || '',
      definition: field.description || '',
      status: 'Pending',
      lastUploaded: null,
      vectorCount: 0,
      namespace: 'data-fields'
    }));
  } catch (error) {
    console.error('Error getting all data fields:', error);
    return [];
  }
}

// Function to get all data fields
export async function getDataFields(): Promise<DataField[]> {
  try {
    // Get the JWT token
    const jwtToken = localStorage.getItem('worldquant_jwt');
    
    if (!jwtToken) {
      return [];
    }
    
    const response = await fetch('/api/data-fields', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        jwtToken,
        dataset: 'fundamental6',
        limit: '20',
        instrumentType: 'EQUITY',
        region: 'USA',
        universe: 'TOP3000',
        delay: '1',
        offset: '0',
      }),
    });
    
    const data = await response.json();
    
    if (data) {
      return data;
    } else {
      console.error('Error getting data fields:', data.error);
      return [];
    }
  } catch (error) {
    console.error('Error getting data fields:', error);
    return [];
  }
}

// Function to get all Pinecone indexes
export async function getPineconeIndexes(): Promise<PineconeIndex[]> {
  try {
    const response = await fetch('/api/pinecone/indexes');
    console.log('Pinecone indexes response:', response);
    const data = await response.json();
    console.log('Pinecone indexes data:', data);

    if (data.success) {
      // Log the data for debugging
      console.log('Pinecone indexes data:', data.indexes);
      
      // Map the data to ensure it matches our interface
      return data.indexes.map((index: any) => ({
        id: index.name || 'unknown',
        dimension: index.dimension || 0,
        metric: index.metric || 'cosine',
        pods: 1, // Default value as this might not be available in the API
        replicas: 1, // Default value as this might not be available in the API
        podType: 'p1.x1', // Default value as this might not be available in the API
        status: index.status?.state || 'Unknown',
        vectorCount: 0, // This will be updated when we get the vector count
        lastUpdated: new Date().toISOString(),
        deletion_protection: index.deletionProtection === 'enabled' ? 'enabled' : 'disabled',
        tags: {}, // Default empty object
        vector_type: index.vectorType || 'dense',
        host: index.host || '',
        spec: {
          serverless: index.spec?.serverless ? {
            cloud: index.spec.serverless.cloud || '',
            region: index.spec.serverless.region || ''
          } : undefined
        }
      }));
    } else {
      console.error('Error getting Pinecone indexes:', data.error);
      return [];
    }
  } catch (error) {
    console.error('Error getting Pinecone indexes:', error);
    return [];
  }
}

// Function to describe a specific Pinecone index
export async function describePineconeIndex(indexName: string): Promise<PineconeIndex | null> {
  try {
    const response = await fetch('/api/pinecone/indexes', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        operation: 'describe',
        indexName,
      }),
    });
    
    const data = await response.json();
    
    if (data.success) {
      return data.index;
    } else {
      console.error('Error describing index:', data.error);
      return null;
    }
  } catch (error) {
    console.error('Error describing index:', error);
    return null;
  }
}

// Function to delete a Pinecone index
export async function deletePineconeIndex(indexName: string): Promise<{ success: boolean; error?: any }> {
  try {
    const response = await fetch('/api/pinecone/indexes', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        operation: 'delete',
        indexName,
      }),
    });
    
    const data = await response.json();
    
    if (data.success) {
      return { success: true };
    } else {
      return { success: false, error: data.error };
    }
  } catch (error) {
    console.error('Error deleting index:', error);
    return { success: false, error };
  }
}

// Function to configure a Pinecone index
export async function configurePineconeIndex(
  indexName: string, 
  config: {
    deletion_protection?: 'enabled' | 'disabled';
    tags?: Record<string, string>;
  }
): Promise<{ success: boolean; error?: any }> {
  try {
    const response = await fetch('/api/pinecone/indexes', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        operation: 'configure',
        indexName,
        config,
      }),
    });
    
    const data = await response.json();
    
    if (data.success) {
      return { success: true };
    } else {
      return { success: false, error: data.error };
    }
  } catch (error) {
    console.error('Error configuring index:', error);
    return { success: false, error };
  }
}

// Function to create a new Pinecone index
export async function createPineconeIndex(
  indexName: string,
  dimension: number,
  metric: string = 'cosine',
  deletion_protection: 'enabled' | 'disabled' = 'disabled',
  tags?: Record<string, string>
): Promise<{ success: boolean; error?: any }> {
  try {
    const response = await fetch('/api/pinecone/indexes', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        operation: 'create',
        indexName,
        config: {
          dimension,
          metric,
          deletion_protection,
          tags,
        },
      }),
    });
    
    const data = await response.json();
    
    if (data.success) {
      return { success: true };
    } else {
      return { success: false, error: data.error };
    }
  } catch (error) {
    console.error('Error creating index:', error);
    return { success: false, error };
  }
}

// Function to upload operator data to Pinecone
export async function uploadOperatorToPinecone(operatorId: string): Promise<{ success: boolean; error?: any }> {
  try {
    const response = await fetch('/api/pinecone/operators', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        operatorId,
      }),
    });
    
    const data = await response.json();
    
    if (data.success) {
      return { success: true };
    } else {
      return { success: false, error: data.error };
    }
  } catch (error) {
    console.error('Error uploading operator to Pinecone:', error);
    return { success: false, error };
  }
}

// Function to upload data field data to Pinecone
export async function uploadDataFieldToPinecone(dataFieldId: string): Promise<{ success: boolean; error?: any }> {
  try {
    const response = await fetch('/api/pinecone/data-fields', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        dataFieldId,
      }),
    });
    
    const data = await response.json();
    
    if (data.success) {
      return { success: true };
    } else {
      return { success: false, error: data.error };
    }
  } catch (error) {
    console.error('Error uploading data field to Pinecone:', error);
    return { success: false, error };
  }
}

// Function to query similar operators
export async function querySimilarOperators(
  queryVector: number[],
  topK: number = 5,
  filter?: any
): Promise<{ success: boolean; results?: any[]; error?: any }> {
  try {
    const response = await fetch('/api/pinecone', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        operation: 'query',
        namespace: 'operators',
        data: {
          vector: queryVector,
          topK,
          filter,
        },
      }),
    });
    
    const data = await response.json();
    
    if (data.success && data.response) {
      return {
        success: true,
        results: data.response.matches?.map((match: any) => ({
          id: match.id,
          name: match.metadata?.name || 'Unknown',
          category: match.metadata?.category || 'Unknown',
          score: match.score || 0
        })) || []
      };
    } else {
      return { success: false, error: data.error };
    }
  } catch (error) {
    console.error('Error querying similar operators:', error);
    return { success: false, error };
  }
}

// Function to query similar data fields
export async function querySimilarDataFields(
  queryVector: number[],
  topK: number = 5,
  filter?: any
): Promise<{ success: boolean; results?: any[]; error?: any }> {
  try {
    const response = await fetch('/api/pinecone', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        operation: 'query',
        namespace: 'data_fields',
        data: {
          vector: queryVector,
          topK,
          filter,
        },
      }),
    });
    
    const data = await response.json();
    
    if (data.success && data.response) {
      return {
        success: true,
        results: data.response.matches?.map((match: any) => ({
          id: match.id,
          name: match.metadata?.name || 'Unknown',
          category: match.metadata?.category || 'Unknown',
          score: match.score || 0
        })) || []
      };
    } else {
      return { success: false, error: data.error };
    }
  } catch (error) {
    console.error('Error querying similar data fields:', error);
    return { success: false, error };
  }
}

export default {
  getOperators,
  getDataFields,
  getPineconeIndexes,
  describePineconeIndex,
  deletePineconeIndex,
  configurePineconeIndex,
  createPineconeIndex,
  uploadOperatorToPinecone,
  uploadDataFieldToPinecone,
  querySimilarOperators,
  querySimilarDataFields
}; 