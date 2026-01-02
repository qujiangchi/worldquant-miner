// Utility functions for Pinecone operations

import { Pinecone } from '@pinecone-database/pinecone';

// Initialize the Pinecone client
const pc = new Pinecone({
  apiKey: process.env.PINECONE_API_KEY || '',
});

// Get the index
const index = pc.index(process.env.PINECONE_INDEX_NAME || 'worldquant-miner');

// Type definitions for Pinecone inference API
export interface DenseEmbedding {
  vectorType: "dense";
  embedding: number[];
}

export interface SparseEmbedding {
  vectorType: "sparse";
  indices: number[];
  values: number[];
}

export type Embedding = DenseEmbedding | SparseEmbedding;

export interface EmbeddingsResponse {
  data: Embedding[];
  model: string;
  usage: {
    total_tokens: number;
  };
}

// Function to generate a random vector for testing
export function generateRandomVector(dimension: number): number[] {
  return Array.from({ length: dimension }, () => (Math.random() * 2) - 1);
}

// Function to format metadata for vectors
export function formatMetadata(data: Record<string, any>): Record<string, any> {
  // Remove any undefined or null values
  const cleanData: Record<string, any> = {};
  for (const [key, value] of Object.entries(data)) {
    if (value !== undefined && value !== null) {
      cleanData[key] = value;
    }
  }
  return cleanData;
}

// Function to validate vector data
export function validateVectorData(vector: number[]): boolean {
  if (!Array.isArray(vector)) {
    return false;
  }
  
  // Check if all elements are numbers
  if (!vector.every(val => typeof val === 'number')) {
    return false;
  }
  
  // Check if vector is not empty
  if (vector.length === 0) {
    return false;
  }
  
  return true;
}

// Function to validate metadata
export function validateMetadata(metadata: Record<string, any>): boolean {
  if (typeof metadata !== 'object' || metadata === null) {
    return false;
  }
  
  // Check if all values are of supported types
  for (const value of Object.values(metadata)) {
    if (
      typeof value !== 'string' &&
      typeof value !== 'number' &&
      typeof value !== 'boolean' &&
      !Array.isArray(value) &&
      !(typeof value === 'object' && value !== null)
    ) {
      return false;
    }
  }
  
  return true;
}

// Function to chunk arrays for batch processing
export function chunkArray<T>(array: T[], size: number): T[][] {
  const chunks: T[][] = [];
  for (let i = 0; i < array.length; i += size) {
    chunks.push(array.slice(i, i + size));
  }
  return chunks;
}

// Function to sleep for a specified time
export function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

// Retry function with exponential backoff
export async function retryWithBackoff<T>(
  operation: () => Promise<T>,
  maxRetries: number = 3,
  initialDelay: number = 1000
): Promise<T> {
  let retries = 0;
  let delay = initialDelay;

  while (true) {
    try {
      return await operation();
    } catch (error) {
      retries++;
      if (retries >= maxRetries) {
        throw error;
      }
      await new Promise(resolve => setTimeout(resolve, delay));
      delay *= 2;
    }
  }
}

export { pc, index }; 