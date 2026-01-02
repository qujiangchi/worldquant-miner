'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Button } from '@/components/ui/button';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Loader2, Database, BookOpen, Settings, Server, Upload } from 'lucide-react';
import { getStoredJWT } from '@/lib/auth';
import PineconeDemo from '@/components/PineconeDemo';
import PineconeDashboard from '@/components/PineconeDashboard';
import PineconeUploader from '@/components/PineconeUploader';

export default function PineconePage() {
  const router = useRouter();
  const [isAuthenticated, setIsAuthenticated] = useState<boolean | null>(null);
  const [activeTab, setActiveTab] = useState('dashboard');

  useEffect(() => {
    const checkAuth = async () => {
      try {
        const token = getStoredJWT();
        if (!token) {
          setIsAuthenticated(false);
          return;
        }
        
        // Simple validation - in a real app, you would verify the token with your backend
        setIsAuthenticated(true);
      } catch (error) {
        console.error('Authentication error:', error);
        setIsAuthenticated(false);
        localStorage.removeItem('jwt_token');
        router.push('/login');
      }
    };
    
    checkAuth();
  }, [router]);

  if (isAuthenticated === null) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <Loader2 className="h-8 w-8 animate-spin text-primary" />
      </div>
    );
  }

  if (!isAuthenticated) {
    return null; // Router will handle redirect
  }

  return (
    <div className="container mx-auto py-8 space-y-8">
      <div className="flex flex-col space-y-2">
        <h1 className="text-3xl font-bold">Pinecone Vector Database</h1>
        <p className="text-muted-foreground">
          Manage and monitor your vector data in Pinecone
        </p>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
        <TabsList className="grid grid-cols-5 w-full max-w-md">
          <TabsTrigger value="dashboard" className="flex items-center gap-2">
            <Database className="h-4 w-4" />
            Dashboard
          </TabsTrigger>
          <TabsTrigger value="demo" className="flex items-center gap-2">
            <BookOpen className="h-4 w-4" />
            Demo
          </TabsTrigger>
          <TabsTrigger value="indexes" className="flex items-center gap-2">
            <Server className="h-4 w-4" />
            Indexes
          </TabsTrigger>
          <TabsTrigger value="uploader" className="flex items-center gap-2">
            <Upload className="h-4 w-4" />
            Uploader
          </TabsTrigger>
          <TabsTrigger value="settings" className="flex items-center gap-2">
            <Settings className="h-4 w-4" />
            Settings
          </TabsTrigger>
        </TabsList>

        <TabsContent value="dashboard" className="space-y-6">
          <PineconeDashboard />
        </TabsContent>

        <TabsContent value="demo" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Pinecone Demo</CardTitle>
              <CardDescription>
                Experiment with Pinecone vector operations
              </CardDescription>
            </CardHeader>
            <CardContent>
              <PineconeDemo />
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="indexes" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Pinecone Indexes</CardTitle>
              <CardDescription>
                Manage your Pinecone vector database indexes
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <Alert>
                <AlertTitle>Index Management</AlertTitle>
                <AlertDescription>
                  The Indexes tab in the dashboard provides a comprehensive interface for managing your Pinecone indexes. You can:
                </AlertDescription>
                <ul className="list-disc pl-6 mt-2 space-y-1">
                  <li>Create new indexes with custom dimensions and metrics</li>
                  <li>Configure deletion protection to prevent accidental deletion</li>
                  <li>Add and manage tags for better organization</li>
                  <li>Delete indexes when they are no longer needed</li>
                  <li>Monitor index status, vector counts, and other metrics</li>
                </ul>
              </Alert>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <Card>
                  <CardHeader>
                    <CardTitle>Index Creation</CardTitle>
                    <CardDescription>Create new vector indexes</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <p className="text-sm text-muted-foreground">
                      Create new Pinecone indexes with custom dimensions, metrics, and settings. 
                      Specify deletion protection and tags for better organization.
                    </p>
                  </CardContent>
                </Card>
                
                <Card>
                  <CardHeader>
                    <CardTitle>Index Configuration</CardTitle>
                    <CardDescription>Configure existing indexes</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <p className="text-sm text-muted-foreground">
                      Modify settings for existing indexes, including enabling/disabling deletion protection 
                      and managing tags for better organization.
                    </p>
                  </CardContent>
                </Card>
              </div>
              
              <Button 
                onClick={() => setActiveTab('dashboard')} 
                className="mt-4"
              >
                Go to Dashboard
              </Button>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="settings" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Pinecone Settings</CardTitle>
              <CardDescription>
                Configure your Pinecone integration
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Alert>
                <AlertTitle>Coming Soon</AlertTitle>
                <AlertDescription>
                  Settings configuration will be available in a future update.
                </AlertDescription>
              </Alert>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="uploader" className="space-y-6">
          <PineconeUploader />
        </TabsContent>
      </Tabs>
    </div>
  );
} 