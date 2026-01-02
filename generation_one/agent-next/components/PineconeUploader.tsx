'use client';

import React, { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { RefreshCw, Upload, AlertCircle, CheckCircle, Clock, XCircle } from 'lucide-react';
import { 
  getOperators, 
  getAllDataFields, 
  uploadOperatorToPinecone, 
  uploadDataFieldToPinecone,
  Operator,
  DataField
} from '@/lib/pinecone-operators';

export default function PineconeUploader() {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [uploadProgress, setUploadProgress] = useState<number>(0);
  const [uploadStatus, setUploadStatus] = useState<'idle' | 'in-progress' | 'completed' | 'error'>('idle');
  const [uploadMessage, setUploadMessage] = useState<string>('');
  const [stats, setStats] = useState<{ operators: number; dataFields: number }>({ operators: 0, dataFields: 0 });

  // Function to upload all data to Pinecone
  const handleUploadAll = async () => {
    setIsLoading(true);
    setError(null);
    setUploadStatus('in-progress');
    setUploadMessage('Fetching operators and data fields from WorldQuant API...');
    setUploadProgress(0);
    
    try {
      // Fetch operators and data fields
      setUploadMessage('Fetching operators...');
      const operators = await getOperators();
      
      setUploadMessage('Fetching data fields (this may take a while)...');
      const dataFields = await getAllDataFields();
      
      setStats({ operators: operators.length, dataFields: dataFields.length });
      
      if (operators.length === 0 && dataFields.length === 0) {
        setError('No operators or data fields found. Please check your WorldQuant API connection.');
        setUploadStatus('error');
        setIsLoading(false);
        return;
      }
      
      setUploadMessage(`Found ${operators.length} operators and ${dataFields.length} data fields. Starting upload...`);
      
      // Upload operators
      let completedOperators = 0;
      let completedDataFields = 0;
      const totalItems = operators.length + dataFields.length;
      
      // Upload operators
      for (const operator of operators) {
        setUploadMessage(`Uploading operator: ${operator.name}...`);
        const result = await fetch('/api/pinecone/operators', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            operatorId: operator.name,
            operatorInfo: {
              name: operator.name || operator.id,
              category: operator.category || 'Uncategorized',
              description: operator.description || `Operator: ${operator.id}`
            }
          }),
        });
        
        completedOperators++;
        setUploadProgress(Math.floor((completedOperators + completedDataFields) / totalItems * 100));
        
        if (!result.ok) {
          const errorData = await result.json();
          console.error(`Failed to upload operator ${operator.name}:`, errorData.error);
        }
      }
      
      // Upload data fields
      for (const field of dataFields) {
        setUploadMessage(`Uploading data field: ${field.name}...`);
        const result = await fetch('/api/pinecone/data-fields', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            dataFieldId: field.id,
            dataFieldInfo: {
              name: field.name || field.id,
              category: field.category || 'Uncategorized',
              description: field.description || field.definition || `Data field: ${field.id}`
            }
          }),
        });
        
        completedDataFields++;
        setUploadProgress(Math.floor((completedOperators + completedDataFields) / totalItems * 100));
        
        if (!result.ok) {
          const errorData = await result.json();
          console.error(`Failed to upload data field ${field.name}:`, errorData.error);
        }
      }
      
      setUploadStatus('completed');
      setUploadMessage(`Upload completed! Successfully processed ${completedOperators} operators and ${completedDataFields} data fields.`);
      
    } catch (err) {
      console.error('Error uploading data:', err);
      setError('Failed to upload data. Please try again.');
      setUploadStatus('error');
    } finally {
      setIsLoading(false);
    }
  };

  // Get status badge component
  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'completed':
        return <div className="flex items-center text-green-500"><CheckCircle className="w-4 h-4 mr-1" /> Completed</div>;
      case 'in-progress':
        return <div className="flex items-center text-blue-500"><Clock className="w-4 h-4 mr-1" /> In Progress</div>;
      case 'error':
        return <div className="flex items-center text-red-500"><XCircle className="w-4 h-4 mr-1" /> Error</div>;
      default:
        return <div className="flex items-center text-gray-500"><Clock className="w-4 h-4 mr-1" /> Ready</div>;
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h2 className="text-2xl font-bold">WorldQuant Data Uploader</h2>
      </div>

      {error && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertTitle>Error</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      <Card>
        <CardHeader>
          <CardTitle>Upload WorldQuant Data to Pinecone</CardTitle>
          <CardDescription>
            Upload operators and data fields from WorldQuant to Pinecone vector database
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="space-y-4">
            <div className="flex justify-between items-center">
              <div>
                <h3 className="text-lg font-medium">Status</h3>
                <p className="text-sm text-muted-foreground">
                  {uploadStatus === 'idle' && 'Ready to upload data'}
                  {uploadStatus === 'in-progress' && 'Uploading data to Pinecone...'}
                  {uploadStatus === 'completed' && `Upload completed! Processed ${stats.operators} operators and ${stats.dataFields} data fields.`}
                  {uploadStatus === 'error' && 'Error uploading data'}
                </p>
              </div>
              <div>
                {getStatusBadge(uploadStatus)}
              </div>
            </div>
            
            {uploadStatus === 'in-progress' && (
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span>Progress</span>
                  <span>{uploadProgress}%</span>
                </div>
                <Progress value={uploadProgress} />
                <p className="text-sm text-muted-foreground">{uploadMessage}</p>
              </div>
            )}
            
            {uploadStatus === 'completed' && (
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span>Progress</span>
                  <span>100%</span>
                </div>
                <Progress value={100} />
                <p className="text-sm text-muted-foreground">{uploadMessage}</p>
              </div>
            )}
          </div>
          
          <div className="flex justify-end">
            <Button 
              onClick={handleUploadAll} 
              disabled={isLoading || uploadStatus === 'in-progress'}
              className="w-full sm:w-auto"
            >
              {isLoading ? (
                <>
                  <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                  Uploading...
                </>
              ) : (
                <>
                  <Upload className="w-4 h-4 mr-2" />
                  Upload All Data
                </>
              )}
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
} 