'use client';

import { useState } from 'react';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '../../../components/ui/card';
import { Button } from '../../../components/ui/button';
import { Loader2, AlertCircle } from 'lucide-react';
import { Alert, AlertDescription, AlertTitle } from '../../../components/ui/alert';
import pineconeClient from '../../../lib/pinecone-client';

// Example data that follows Pinecone's metadata requirements
const exampleData = [
  {
    count: 2,
    results: [
      {
        id: "asset1",
        description: "US Equity Market Alpha",
        dataset: "US_EQUITY",
        category: "Market",
        subcategory: "Alpha",
        region: "US",
        delay: 0,
        universe: "US_LARGE_CAP",
        type: "SIGNAL",
        coverage: 0.95,
        userCount: 150,
        alphaCount: 25,
        themes: ["Momentum", "Value", "Quality"]
      },
      {
        id: "asset2",
        description: "European Fixed Income Strategy",
        dataset: "EU_FIXED_INCOME",
        category: "Fixed Income",
        subcategory: "Strategy",
        region: "EU",
        delay: 1,
        universe: "EU_IG",
        type: "STRATEGY",
        coverage: 0.85,
        userCount: 75,
        alphaCount: 12,
        themes: ["Duration", "Credit", "Yield"]
      }
    ]
  }
];

export default function PineconeExamplePage() {
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  const handleUpsertExample = async () => {
    try {
      setIsLoading(true);
      setError(null);
      
      // Upsert the example data
      const response = await pineconeClient.upsertVectors('examples', exampleData);
      
      if (response.success) {
        setResult(response);
      } else {
        setError('Failed to upsert vectors: ' + JSON.stringify(response.error));
      }
    } catch (error) {
      console.error('Error upserting example data:', error);
      setError(error instanceof Error ? error.message : 'An error occurred');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="container mx-auto py-8">
      <h1 className="text-3xl font-bold mb-6">Pinecone Integration Example</h1>
      <p className="text-muted-foreground mb-8">
        This example demonstrates how to integrate Pinecone with your application using the WorldQuant Miner.
      </p>

      <Alert className="mb-6">
        <AlertCircle className="h-4 w-4" />
        <AlertTitle>Vector Dimensions</AlertTitle>
        <AlertDescription>
          Your Pinecone index has a fixed dimension (typically 1024 or 1536). If your vectors have a different dimension,
          they will be automatically adjusted to match the index dimension. Vectors with more dimensions will be truncated,
          and vectors with fewer dimensions will be padded with random values.
        </AlertDescription>
      </Alert>

      <Alert className="mb-6">
        <AlertCircle className="h-4 w-4" />
        <AlertTitle>Vector Values</AlertTitle>
        <AlertDescription>
          Pinecone requires vectors to contain at least one non-zero value. If you don't provide vector values,
          random values will be generated automatically. For best results, use normalized vectors with values between -1 and 1.
        </AlertDescription>
      </Alert>

      <Alert className="mb-6">
        <AlertCircle className="h-4 w-4" />
        <AlertTitle>Metadata Format</AlertTitle>
        <AlertDescription>
          Pinecone requires metadata values to be strings, numbers, booleans, or lists of strings. 
          Objects or arrays of objects are not allowed in metadata.
        </AlertDescription>
      </Alert>

      <Alert className="mb-6">
        <AlertTitle>Example Data Format</AlertTitle>
        <AlertDescription>
          This example demonstrates how to structure data for Pinecone:
          <ul className="list-disc pl-6 mt-2">
            <li>Each item has a unique ID</li>
            <li>Metadata values are strings, numbers, booleans, or arrays of strings</li>
            <li>Complex objects are automatically converted to strings</li>
          </ul>
          <p className="mt-2">
            The example data includes various metadata fields that demonstrate proper formatting.
          </p>
        </AlertDescription>
      </Alert>

      <Card className="mb-8">
        <CardHeader>
          <CardTitle>Upsert Example Data</CardTitle>
          <CardDescription>
            Click the button below to upsert the example data to your Pinecone index.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <pre className="bg-muted p-4 rounded-md overflow-auto max-h-60 text-xs">
            {JSON.stringify(exampleData, null, 2)}
          </pre>
        </CardContent>
        <CardFooter>
          <Button onClick={handleUpsertExample} disabled={isLoading}>
            {isLoading ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Upserting...
              </>
            ) : (
              'Upsert Example Data'
            )}
          </Button>
        </CardFooter>
      </Card>

      {error && (
        <Card className="mb-8 border-destructive">
          <CardHeader>
            <CardTitle className="text-destructive">Error</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-destructive">{error}</p>
          </CardContent>
        </Card>
      )}

      {result && (
        <Card>
          <CardHeader>
            <CardTitle>Result</CardTitle>
          </CardHeader>
          <CardContent>
            <pre className="bg-muted p-4 rounded-md overflow-auto max-h-60 text-xs">
              {JSON.stringify(result, null, 2)}
            </pre>
          </CardContent>
        </Card>
      )}
    </div>
  );
} 