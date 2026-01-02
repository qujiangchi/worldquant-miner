'use client';

import { useState } from 'react';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Textarea } from './ui/textarea';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from './ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { Loader2, Info, AlertCircle } from 'lucide-react';
import pineconeClient from '../lib/pinecone-client';
import { Alert, AlertDescription, AlertTitle } from './ui/alert';
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from './ui/accordion';

export default function PineconeDemo() {
  const [namespace, setNamespace] = useState('ns1');
  const [vectors, setVectors] = useState('');
  const [queryVector, setQueryVector] = useState('');
  const [topK, setTopK] = useState('5');
  const [filter, setFilter] = useState('');
  const [result, setResult] = useState<any>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState('upsert');

  // Example data for different operations
  const exampleData = {
    upsert: `[
  {
    "id": "vec1",
    "values": [0.1, 0.2, 0.3, 0.4, 0.5],
    "metadata": {
      "title": "Example Vector 1",
      "category": "test"
    }
  },
  {
    "id": "vec2",
    "values": [0.2, 0.3, 0.4, 0.5, 0.6],
    "metadata": {
      "title": "Example Vector 2",
      "category": "test"
    }
  }
]`,
    query: `[0.1, 0.2, 0.3, 0.4, 0.5]`,
    filter: `{"category": {"$eq": "test"}}`,
    delete: `["vec1", "vec2"]`
  };

  const handleUpsert = async () => {
    try {
      setIsLoading(true);
      setError(null);
      
      // Parse the vectors
      let parsedVectors;
      try {
        parsedVectors = JSON.parse(vectors);
      } catch (e) {
        throw new Error('Invalid JSON format for vectors');
      }
      
      // Validate the vectors
      if (!Array.isArray(parsedVectors)) {
        throw new Error('Vectors must be an array');
      }
      
      // Upsert the vectors
      const response = await pineconeClient.upsertVectors(namespace, parsedVectors);
      setResult(response);
    } catch (error) {
      console.error('Error upserting vectors:', error);
      setError(error instanceof Error ? error.message : 'An error occurred');
    } finally {
      setIsLoading(false);
    }
  };

  const handleQuery = async () => {
    try {
      setIsLoading(true);
      setError(null);
      
      // Parse the query vector
      let parsedVector;
      try {
        parsedVector = JSON.parse(queryVector);
      } catch (e) {
        throw new Error('Invalid JSON format for query vector');
      }
      
      // Validate the query vector
      if (!Array.isArray(parsedVector)) {
        throw new Error('Query vector must be an array');
      }
      
      // Parse the filter if provided
      let parsedFilter = undefined;
      if (filter) {
        try {
          parsedFilter = JSON.parse(filter);
        } catch (e) {
          throw new Error('Invalid JSON format for filter');
        }
      }
      
      // Query the vectors
      const response = await pineconeClient.queryVectors(
        namespace,
        parsedVector,
        parseInt(topK),
        parsedFilter
      );
      setResult(response);
    } catch (error) {
      console.error('Error querying vectors:', error);
      setError(error instanceof Error ? error.message : 'An error occurred');
    } finally {
      setIsLoading(false);
    }
  };

  const handleDelete = async () => {
    try {
      setIsLoading(true);
      setError(null);
      
      // Parse the vector IDs
      let parsedIds;
      try {
        parsedIds = JSON.parse(vectors);
      } catch (e) {
        throw new Error('Invalid JSON format for vector IDs');
      }
      
      // Validate the vector IDs
      if (!Array.isArray(parsedIds)) {
        throw new Error('Vector IDs must be an array');
      }
      
      // Delete the vectors
      const response = await pineconeClient.deleteVectors(namespace, parsedIds);
      setResult(response);
    } catch (error) {
      console.error('Error deleting vectors:', error);
      setError(error instanceof Error ? error.message : 'An error occurred');
    } finally {
      setIsLoading(false);
    }
  };

  const handleDeleteAll = async () => {
    try {
      setIsLoading(true);
      setError(null);
      
      // Delete all vectors
      const response = await pineconeClient.deleteAllVectors(namespace);
      setResult(response);
    } catch (error) {
      console.error('Error deleting all vectors:', error);
      setError(error instanceof Error ? error.message : 'An error occurred');
    } finally {
      setIsLoading(false);
    }
  };

  const loadExample = (type: string) => {
    switch (type) {
      case 'upsert':
        setVectors(exampleData.upsert);
        break;
      case 'query':
        setQueryVector(exampleData.query);
        break;
      case 'filter':
        setFilter(exampleData.filter);
        break;
      case 'delete':
        setVectors(exampleData.delete);
        break;
    }
  };

  return (
    <Card className="w-full max-w-4xl mx-auto">
      <CardHeader>
        <CardTitle>Pinecone Vector Database Demo</CardTitle>
        <CardDescription>
          Interact with your Pinecone vector database. You can upsert vectors, query them, and delete them.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <Alert className="mb-6">
          <Info className="h-4 w-4" />
          <AlertTitle>About Pinecone</AlertTitle>
          <AlertDescription>
            Pinecone is a vector database that allows you to store and query vector embeddings. 
            It's perfect for semantic search, recommendation systems, and other AI applications.
          </AlertDescription>
        </Alert>

        <Alert className="mb-6">
          <AlertCircle className="h-4 w-4" />
          <AlertTitle>Important: Metadata Format Requirements</AlertTitle>
          <AlertDescription>
            Pinecone requires metadata values to be one of the following types:
            <ul className="list-disc pl-6 mt-2">
              <li>Strings</li>
              <li>Numbers</li>
              <li>Booleans</li>
              <li>Arrays of strings</li>
            </ul>
            <p className="mt-2">
              Objects, arrays of objects, or other complex types will be automatically converted to strings.
              For best results, ensure your metadata follows these requirements.
            </p>
          </AlertDescription>
        </Alert>

        <Alert className="mb-6">
          <AlertCircle className="h-4 w-4" />
          <AlertTitle>Vector Dimension Requirements</AlertTitle>
          <AlertDescription>
            The Pinecone index has a fixed dimension (typically 1024 or 1536). If your vectors have:
            <ul className="list-disc pl-6 mt-2">
              <li>More dimensions: They will be truncated</li>
              <li>Fewer dimensions: They will be padded with random values</li>
            </ul>
            <p className="mt-2">
              For optimal results, ensure your vectors match the index dimension.
            </p>
          </AlertDescription>
        </Alert>

        <Alert className="mb-6">
          <AlertCircle className="h-4 w-4" />
          <AlertTitle>Vector Value Requirements</AlertTitle>
          <AlertDescription>
            Pinecone requires vectors to contain at least one non-zero value. If your vectors contain only zeros:
            <ul className="list-disc pl-6 mt-2">
              <li>Random values will be generated automatically</li>
              <li>For best results, use normalized vectors with values between -1 and 1</li>
            </ul>
          </AlertDescription>
        </Alert>

        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <TabsList className="grid grid-cols-4 mb-4">
            <TabsTrigger value="upsert">Upsert</TabsTrigger>
            <TabsTrigger value="query">Query</TabsTrigger>
            <TabsTrigger value="delete">Delete</TabsTrigger>
            <TabsTrigger value="deleteAll">Delete All</TabsTrigger>
          </TabsList>
          
          <TabsContent value="upsert">
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium mb-1">Namespace</label>
                <Input
                  value={namespace}
                  onChange={(e) => setNamespace(e.target.value)}
                  placeholder="Enter namespace"
                />
                <p className="text-xs text-muted-foreground mt-1">
                  Namespaces help you organize your vectors. They're like folders in a file system.
                </p>
              </div>
              <div>
                <label className="block text-sm font-medium mb-1">Vectors (JSON)</label>
                <Textarea
                  value={vectors}
                  onChange={(e) => setVectors(e.target.value)}
                  placeholder="Enter vectors in JSON format"
                  className="min-h-[200px] font-mono text-sm"
                />
                <div className="flex justify-between items-center mt-1">
                  <p className="text-xs text-muted-foreground">
                    Format: Array of objects with id, values, and metadata. 
                    For data with count and results structure, it will be automatically transformed.
                  </p>
                  <Button variant="outline" size="sm" onClick={() => loadExample('upsert')}>
                    Load Example
                  </Button>
                </div>
              </div>
              <Button onClick={handleUpsert} disabled={isLoading}>
                {isLoading ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Upserting...
                  </>
                ) : (
                  'Upsert Vectors'
                )}
              </Button>
            </div>
          </TabsContent>
          
          <TabsContent value="query">
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium mb-1">Namespace</label>
                <Input
                  value={namespace}
                  onChange={(e) => setNamespace(e.target.value)}
                  placeholder="Enter namespace"
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-1">Query Vector (JSON array)</label>
                <Textarea
                  value={queryVector}
                  onChange={(e) => setQueryVector(e.target.value)}
                  placeholder="[0.1, 0.2, 0.3, 0.4, 0.5]"
                  className="min-h-[100px] font-mono text-sm"
                />
                <div className="flex justify-between items-center mt-1">
                  <p className="text-xs text-muted-foreground">
                    The vector to query against. Must be an array of numbers with the same dimension as your index.
                  </p>
                  <Button variant="outline" size="sm" onClick={() => loadExample('query')}>
                    Load Example
                  </Button>
                </div>
              </div>
              <div>
                <label className="block text-sm font-medium mb-1">Top K</label>
                <Input
                  type="number"
                  value={topK}
                  onChange={(e) => setTopK(e.target.value)}
                  placeholder="5"
                  min="1"
                  max="100"
                />
                <p className="text-xs text-muted-foreground mt-1">
                  The number of results to return. Higher values return more results but may be slower.
                </p>
              </div>
              <div>
                <label className="block text-sm font-medium mb-1">Filter (JSON object, optional)</label>
                <Textarea
                  value={filter}
                  onChange={(e) => setFilter(e.target.value)}
                  placeholder='{"category": {"$eq": "test"}}'
                  className="min-h-[100px] font-mono text-sm"
                />
                <div className="flex justify-between items-center mt-1">
                  <p className="text-xs text-muted-foreground">
                    Filter results based on metadata. Uses MongoDB-style query syntax.
                  </p>
                  <Button variant="outline" size="sm" onClick={() => loadExample('filter')}>
                    Load Example
                  </Button>
                </div>
              </div>
              <Button onClick={handleQuery} disabled={isLoading}>
                {isLoading ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Querying...
                  </>
                ) : (
                  'Query Vectors'
                )}
              </Button>
            </div>
          </TabsContent>
          
          <TabsContent value="delete">
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium mb-1">Namespace</label>
                <Input
                  value={namespace}
                  onChange={(e) => setNamespace(e.target.value)}
                  placeholder="Enter namespace"
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-1">Vector IDs to Delete (JSON array)</label>
                <Textarea
                  value={vectors}
                  onChange={(e) => setVectors(e.target.value)}
                  placeholder='["vec1", "vec2"]'
                  className="min-h-[100px] font-mono text-sm"
                />
                <div className="flex justify-between items-center mt-1">
                  <p className="text-xs text-muted-foreground">
                    Array of vector IDs to delete from the namespace.
                  </p>
                  <Button variant="outline" size="sm" onClick={() => loadExample('delete')}>
                    Load Example
                  </Button>
                </div>
              </div>
              <Button onClick={handleDelete} disabled={isLoading}>
                {isLoading ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Deleting...
                  </>
                ) : (
                  'Delete Vectors'
                )}
              </Button>
            </div>
          </TabsContent>
          
          <TabsContent value="deleteAll">
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium mb-1">Namespace</label>
                <Input
                  value={namespace}
                  onChange={(e) => setNamespace(e.target.value)}
                  placeholder="Enter namespace"
                />
              </div>
              <Alert variant="destructive" className="mb-4">
                <AlertCircle className="h-4 w-4" />
                <AlertTitle>Warning</AlertTitle>
                <AlertDescription>
                  This will delete all vectors in the specified namespace. This action cannot be undone.
                </AlertDescription>
              </Alert>
              <Button variant="destructive" onClick={handleDeleteAll} disabled={isLoading}>
                {isLoading ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Deleting All...
                  </>
                ) : (
                  'Delete All Vectors'
                )}
              </Button>
            </div>
          </TabsContent>
        </Tabs>
      </CardContent>
      
      {error && (
        <Card className="mt-6 border-destructive">
          <CardHeader>
            <CardTitle className="text-destructive">Error</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-destructive">{error}</p>
          </CardContent>
        </Card>
      )}
      
      {result && (
        <Card className="mt-6">
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

      <Accordion type="single" collapsible className="mt-6">
        <AccordionItem value="pinecone-guide">
          <AccordionTrigger>Pinecone Guide for Beginners</AccordionTrigger>
          <AccordionContent>
            <div className="space-y-4 text-sm">
              <h3 className="font-medium">What is Pinecone?</h3>
              <p>
                Pinecone is a vector database that allows you to store and query vector embeddings. 
                Vector embeddings are numerical representations of data (like text, images, or other structured data) 
                that capture their semantic meaning.
              </p>
              
              <h3 className="font-medium">Key Concepts</h3>
              <ul className="list-disc pl-5 space-y-1">
                <li><strong>Index:</strong> A collection of vectors with the same dimensionality.</li>
                <li><strong>Namespace:</strong> A partition within an index, like a folder in a file system.</li>
                <li><strong>Vector:</strong> A numerical representation of data with a specific dimensionality.</li>
                <li><strong>Metadata:</strong> Additional information associated with a vector.</li>
              </ul>
              
              <h3 className="font-medium">Common Operations</h3>
              <ul className="list-disc pl-5 space-y-1">
                <li><strong>Upsert:</strong> Add or update vectors in a namespace.</li>
                <li><strong>Query:</strong> Find the most similar vectors to a query vector.</li>
                <li><strong>Delete:</strong> Remove specific vectors from a namespace.</li>
                <li><strong>Delete All:</strong> Remove all vectors from a namespace.</li>
              </ul>
              
              <h3 className="font-medium">Best Practices</h3>
              <ul className="list-disc pl-5 space-y-1">
                <li>Use namespaces to organize your vectors by category, user, or other logical grouping.</li>
                <li>Include relevant metadata with your vectors to enable filtering during queries.</li>
                <li>Use appropriate vector dimensions based on your embedding model.</li>
                <li>Consider using filters to narrow down query results for better performance.</li>
              </ul>
              
              <h3 className="font-medium">Common Use Cases</h3>
              <ul className="list-disc pl-5 space-y-1">
                <li>Semantic search</li>
                <li>Recommendation systems</li>
                <li>Similarity matching</li>
                <li>Anomaly detection</li>
                <li>Content personalization</li>
              </ul>
            </div>
          </AccordionContent>
        </AccordionItem>
      </Accordion>
    </Card>
  );
} 