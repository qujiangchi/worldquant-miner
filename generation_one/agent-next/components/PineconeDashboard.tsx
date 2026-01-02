'use client';

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Switch } from '@/components/ui/switch';
import { RefreshCw, Upload, Search, AlertCircle, CheckCircle, Clock, XCircle, Plus, Trash2, Settings, Tag, Database } from 'lucide-react';
import { 
  getOperators, 
  getDataFields, 
  getPineconeIndexes, 
  describePineconeIndex,
  deletePineconeIndex,
  configurePineconeIndex,
  createPineconeIndex,
  uploadOperatorToPinecone, 
  uploadDataFieldToPinecone,
  Operator,
  DataField,
  PineconeIndex
} from '@/lib/pinecone-operators';

export default function PineconeDashboard() {
  const [activeTab, setActiveTab] = useState('overview');
  const [indexes, setIndexes] = useState<PineconeIndex[]>([]);
  const [operators, setOperators] = useState<Operator[]>([]);
  const [dataFields, setDataFields] = useState<DataField[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [uploadProgress, setUploadProgress] = useState<Record<string, number>>({});
  const [searchQuery, setSearchQuery] = useState('');
  
  // Index management states
  const [selectedIndex, setSelectedIndex] = useState<PineconeIndex | null>(null);
  const [isCreateIndexDialogOpen, setIsCreateIndexDialogOpen] = useState(false);
  const [isConfigureIndexDialogOpen, setIsConfigureIndexDialogOpen] = useState(false);
  const [isDeleteIndexDialogOpen, setIsDeleteIndexDialogOpen] = useState(false);
  const [newIndexName, setNewIndexName] = useState('');
  const [newIndexDimension, setNewIndexDimension] = useState('1536');
  const [newIndexMetric, setNewIndexMetric] = useState('cosine');
  const [newIndexDeletionProtection, setNewIndexDeletionProtection] = useState(false);
  const [newIndexTags, setNewIndexTags] = useState<Record<string, string>>({});
  const [newTagKey, setNewTagKey] = useState('');
  const [newTagValue, setNewTagValue] = useState('');
  const [configureDeletionProtection, setConfigureDeletionProtection] = useState(false);
  const [configureTags, setConfigureTags] = useState<Record<string, string>>({});
  const [configureTagKey, setConfigureTagKey] = useState('');
  const [configureTagValue, setConfigureTagValue] = useState('');

  // Add this state variable at the top with other state variables
  const [selectedIndexDetails, setSelectedIndexDetails] = useState<PineconeIndex | null>(null);
  const [isIndexDetailsOpen, setIsIndexDetailsOpen] = useState(false);

  // Fetch data on component mount
  useEffect(() => {
    refreshData();
  }, []);

  // Function to refresh all data
  const refreshData = async () => {
    setIsLoading(true);
    setError(null);
    
    try {
      console.log('Fetching data from API...');
      
      const [indexesData, operatorsData, dataFieldsData] = await Promise.all([
        getPineconeIndexes(),
        getOperators(),
        getDataFields()
      ]);
      
      console.log('Indexes data from API:', indexesData);
      console.log('Operators data from API:', operatorsData);
      console.log('Data fields data from API:', dataFieldsData);
      
      if (indexesData && Array.isArray(indexesData)) {
        console.log('Setting indexes data:', indexesData);
        setIndexes(indexesData);
      } else {
        console.log('No indexes data available');
        setIndexes([]);
      }
      
      if (operatorsData && Array.isArray(operatorsData)) {
        setOperators(operatorsData);
      } else {
        setOperators([]);
      }
      
      if (dataFieldsData && Array.isArray(dataFieldsData)) {
        setDataFields(dataFieldsData);
      } else {
        setDataFields([]);
      }
    } catch (err) {
      console.error('Error fetching data:', err);
      setError('Failed to fetch data. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  // Function to upload operator to Pinecone
  const handleUploadOperator = async (operatorId: string) => {
    setUploadProgress(prev => ({ ...prev, [operatorId]: 0 }));
    
    try {
      // Simulate progress updates
      const progressInterval = setInterval(() => {
        setUploadProgress(prev => {
          const currentProgress = prev[operatorId] || 0;
          if (currentProgress >= 90) {
            clearInterval(progressInterval);
            return prev;
          }
          return { ...prev, [operatorId]: currentProgress + 10 };
        });
      }, 300);
      
      const operator = operators.find(op => op.id === operatorId);
      if (!operator) {
        throw new Error('Operator not found');
      }

      // Format the operator information
      const operatorInfo = {
        name: operator.name || operatorId,
        category: operator.category || 'Uncategorized',
        description: operator.description || `Operator: ${operatorId}`
      };

      const result = await fetch('/api/pinecone/operators', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          operatorId,
          operatorInfo
        }),
      });

      if (!result.ok) {
        const errorData = await result.json();
        throw new Error(errorData.error || 'Failed to upload operator');
      }
      
      clearInterval(progressInterval);
      setUploadProgress(prev => ({ ...prev, [operatorId]: 100 }));
      
      // Refresh operators to get updated status
      const updatedOperators = await getOperators();
      setOperators(updatedOperators);
    } catch (err) {
      console.error('Error uploading operator:', err);
      setError(`Failed to upload operator: ${err instanceof Error ? err.message : 'Unknown error'}`);
    }
  };

  // Function to upload data field to Pinecone
  const handleUploadDataField = async (dataFieldId: string) => {
    setUploadProgress(prev => ({ ...prev, [dataFieldId]: 0 }));
    
    try {
      // Simulate progress updates
      const progressInterval = setInterval(() => {
        setUploadProgress(prev => {
          const currentProgress = prev[dataFieldId] || 0;
          if (currentProgress >= 90) {
            clearInterval(progressInterval);
            return prev;
          }
          return { ...prev, [dataFieldId]: currentProgress + 10 };
        });
      }, 300);
      
      const dataField = dataFields.find(df => df.id === dataFieldId);
      if (!dataField) {
        throw new Error('Data field not found');
      }

      // Format the data field information
      const dataFieldInfo = {
        name: dataField.name || dataField.id,
        category: dataField.category || 'Uncategorized',
        description: dataField.description || dataField.definition || `Data field: ${dataField.id}`
      };

      const result = await fetch('/api/pinecone/data-fields', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          dataFieldId,
          dataFieldInfo
        }),
      });

      if (!result.ok) {
        const errorData = await result.json();
        throw new Error(errorData.error || 'Failed to upload data field');
      }
      
      clearInterval(progressInterval);
      setUploadProgress(prev => ({ ...prev, [dataFieldId]: 100 }));
      
      // Refresh data fields to get updated status
      const updatedDataFields = await getDataFields();
      setDataFields(updatedDataFields);
    } catch (err) {
      console.error('Error uploading data field:', err);
      setError(`Failed to upload data field: ${err instanceof Error ? err.message : 'Unknown error'}`);
    }
  };

  // Function to handle index selection
  const handleSelectIndex = (index: PineconeIndex) => {
    setSelectedIndex(index);
    setConfigureDeletionProtection(index.deletion_protection === 'enabled');
    setConfigureTags(index.tags || {});
  };

  // Function to create a new index
  const handleCreateIndex = async () => {
    if (!newIndexName || !newIndexDimension) {
      setError('Index name and dimension are required');
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const result = await createPineconeIndex(
        newIndexName,
        parseInt(newIndexDimension),
        newIndexMetric,
        newIndexDeletionProtection ? 'enabled' : 'disabled',
        Object.keys(newIndexTags).length > 0 ? newIndexTags : undefined
      );

      if (result.success) {
        // Reset form
        setNewIndexName('');
        setNewIndexDimension('1536');
        setNewIndexMetric('cosine');
        setNewIndexDeletionProtection(false);
        setNewIndexTags({});
        
        // Close dialog
        setIsCreateIndexDialogOpen(false);
        
        // Refresh data
        await refreshData();
      } else {
        setError(`Failed to create index: ${result.error?.message || 'Unknown error'}`);
      }
    } catch (err) {
      console.error('Error creating index:', err);
      setError('Failed to create index. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  // Function to configure an index
  const handleConfigureIndex = async () => {
    if (!selectedIndex) {
      setError('No index selected');
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const result = await configurePineconeIndex(
        selectedIndex.id,
        {
          deletion_protection: configureDeletionProtection ? 'enabled' : 'disabled',
          tags: Object.keys(configureTags).length > 0 ? configureTags : undefined
        }
      );

      if (result.success) {
        // Close dialog
        setIsConfigureIndexDialogOpen(false);
        
        // Refresh data
        await refreshData();
      } else {
        setError(`Failed to configure index: ${result.error?.message || 'Unknown error'}`);
      }
    } catch (err) {
      console.error('Error configuring index:', err);
      setError('Failed to configure index. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  // Function to delete an index
  const handleDeleteIndex = async () => {
    if (!selectedIndex) {
      setError('No index selected');
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const result = await deletePineconeIndex(selectedIndex.id);

      if (result.success) {
        // Close dialog
        setIsDeleteIndexDialogOpen(false);
        
        // Refresh data
        await refreshData();
      } else {
        setError(`Failed to delete index: ${result.error?.message || 'Unknown error'}`);
      }
    } catch (err) {
      console.error('Error deleting index:', err);
      setError('Failed to delete index. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  // Function to add a new tag to the new index
  const handleAddNewTag = () => {
    if (newTagKey && newTagValue) {
      setNewIndexTags(prev => ({
        ...prev,
        [newTagKey]: newTagValue
      }));
      setNewTagKey('');
      setNewTagValue('');
    }
  };

  // Function to remove a tag from the new index
  const handleRemoveNewTag = (key: string) => {
    setNewIndexTags(prev => {
      const newTags = { ...prev };
      delete newTags[key];
      return newTags;
    });
  };

  // Function to add a tag to the configure index
  const handleAddConfigureTag = () => {
    if (configureTagKey && configureTagValue) {
      setConfigureTags(prev => ({
        ...prev,
        [configureTagKey]: configureTagValue
      }));
      setConfigureTagKey('');
      setConfigureTagValue('');
    }
  };

  // Function to remove a tag from the configure index
  const handleRemoveConfigureTag = (key: string) => {
    setConfigureTags(prev => {
      const newTags = { ...prev };
      delete newTags[key];
      return newTags;
    });
  };

  // Filter operators and data fields based on search query
  const filteredOperators = operators.filter(op => 
    op.name.toLowerCase().includes(searchQuery.toLowerCase()) || 
    op.category.toLowerCase().includes(searchQuery.toLowerCase())
  );
  
  const filteredDataFields = dataFields.filter(df => 
    df.name.toLowerCase().includes(searchQuery.toLowerCase()) || 
    df.category.toLowerCase().includes(searchQuery.toLowerCase())
  );

  // Get status badge component
  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'Uploaded':
        return <Badge variant="default"><CheckCircle className="w-3 h-3 mr-1" /> Uploaded</Badge>;
      case 'In Progress':
        return <Badge variant="secondary"><Clock className="w-3 h-3 mr-1" /> In Progress</Badge>;
      case 'Error':
        return <Badge variant="destructive"><XCircle className="w-3 h-3 mr-1" /> Error</Badge>;
      default:
        return <Badge variant="outline"><Clock className="w-3 h-3 mr-1" /> Pending</Badge>;
    }
  };

  // Add this function with other handler functions
  const handleViewIndexDetails = (index: PineconeIndex) => {
    setSelectedIndexDetails(index);
    setIsIndexDetailsOpen(true);
  };

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h2 className="text-2xl font-bold">Pinecone Dashboard</h2>
        <Button onClick={refreshData} disabled={isLoading}>
          <RefreshCw className={`w-4 h-4 mr-2 ${isLoading ? 'animate-spin' : ''}`} />
          Refresh
        </Button>
      </div>

      {error && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertTitle>Error</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="grid grid-cols-4 w-full">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="operators">Operators</TabsTrigger>
          <TabsTrigger value="data-fields">Data Fields</TabsTrigger>
          <TabsTrigger value="indexes">Indexes</TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <Card>
              <CardHeader>
                <CardTitle>Total Indexes</CardTitle>
                <CardDescription>Pinecone indexes</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="text-3xl font-bold">{indexes.length}</div>
                <div className="text-sm text-muted-foreground mt-1">
                  {indexes.filter(idx => idx.status === 'Ready').length} ready
                </div>
                {indexes.length > 0 && (
                  <div className="text-sm text-muted-foreground mt-1">
                    Total vectors: {indexes.reduce((sum, idx) => sum + idx.vectorCount, 0).toLocaleString()}
                  </div>
                )}
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Total Operators</CardTitle>
                <CardDescription>Alpha operators</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="text-3xl font-bold">{operators.length}</div>
                <div className="text-sm text-muted-foreground mt-1">
                  {operators.filter(op => op.status === 'Uploaded').length} uploaded
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Total Data Fields</CardTitle>
                <CardDescription>Market data fields</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="text-3xl font-bold">{dataFields.length}</div>
                <div className="text-sm text-muted-foreground mt-1">
                  {dataFields.filter(df => df.status === 'Uploaded').length} uploaded
                </div>
              </CardContent>
            </Card>
          </div>

          {indexes.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle>Index Overview</CardTitle>
                <CardDescription>Summary of your Pinecone indexes</CardDescription>
              </CardHeader>
              <CardContent>
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Name</TableHead>
                      <TableHead>Dimension</TableHead>
                      <TableHead>Status</TableHead>
                      <TableHead>Vector Count</TableHead>
                      <TableHead>Last Updated</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {indexes.map((index) => (
                      <TableRow key={index.id}>
                        <TableCell className="font-medium">{index.id}</TableCell>
                        <TableCell>{index.dimension}</TableCell>
                        <TableCell>
                          <Badge variant={index.status === 'Ready' ? 'default' : index.status === 'Initializing' ? 'secondary' : 'outline'}>
                            {index.status}
                          </Badge>
                        </TableCell>
                        <TableCell>{index.vectorCount.toLocaleString()}</TableCell>
                        <TableCell>{new Date(index.lastUpdated).toLocaleString()}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </CardContent>
            </Card>
          )}

          <Card>
            <CardHeader>
              <CardTitle>Upload Status</CardTitle>
              <CardDescription>Current upload progress</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {Object.entries(uploadProgress).map(([id, progress]) => {
                  const item = [...operators, ...dataFields].find(item => item.id === id);
                  if (!item) return null;
                  
                  return (
                    <div key={id} className="space-y-1">
                      <div className="flex justify-between text-sm">
                        <span>{item.name}</span>
                        <span>{progress}%</span>
                      </div>
                      <Progress value={progress} />
                    </div>
                  );
                })}
                
                {Object.keys(uploadProgress).length === 0 && (
                  <p className="text-sm text-muted-foreground">No active uploads</p>
                )}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="operators" className="space-y-4">
          <div className="flex items-center space-x-2">
            <Input
              placeholder="Search operators..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="max-w-sm"
            />
            <Button variant="outline">
              <Search className="w-4 h-4 mr-2" />
              Search
            </Button>
          </div>

          <Card>
            <CardHeader>
              <CardTitle>Operators</CardTitle>
              <CardDescription>Alpha operators in Pinecone</CardDescription>
            </CardHeader>
            <CardContent>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Name</TableHead>
                    <TableHead>Category</TableHead>
                    <TableHead>Status</TableHead>
                    <TableHead>Vector Count</TableHead>
                    <TableHead>Last Uploaded</TableHead>
                    <TableHead>Actions</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {filteredOperators.map((operator) => (
                    <TableRow key={operator.id}>
                      <TableCell className="font-medium">{operator.name}</TableCell>
                      <TableCell>{operator.category}</TableCell>
                      <TableCell>{getStatusBadge(operator.status)}</TableCell>
                      <TableCell>{operator.vectorCount}</TableCell>
                      <TableCell>
                        {operator.lastUploaded 
                          ? new Date(operator.lastUploaded).toLocaleString() 
                          : 'Never'}
                      </TableCell>
                      <TableCell>
                        {operator.status !== 'Uploaded' && (
                          <Button 
                            size="sm" 
                            onClick={() => handleUploadOperator(operator.id)}
                            disabled={operator.status === 'In Progress' || !!uploadProgress[operator.id]}
                          >
                            <Upload className="w-4 h-4 mr-1" />
                            Upload
                          </Button>
                        )}
                        {uploadProgress[operator.id] !== undefined && uploadProgress[operator.id] < 100 && (
                          <Progress value={uploadProgress[operator.id]} className="w-20" />
                        )}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="data-fields" className="space-y-4">
          <div className="flex items-center space-x-2">
            <Input
              placeholder="Search data fields..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="max-w-sm"
            />
            <Button variant="outline">
              <Search className="w-4 h-4 mr-2" />
              Search
            </Button>
          </div>

          <Card>
            <CardHeader>
              <CardTitle>Data Fields</CardTitle>
              <CardDescription>Market data fields in Pinecone</CardDescription>
            </CardHeader>
            <CardContent>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Name</TableHead>
                    <TableHead>Category</TableHead>
                    <TableHead>Status</TableHead>
                    <TableHead>Vector Count</TableHead>
                    <TableHead>Last Uploaded</TableHead>
                    <TableHead>Actions</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {filteredDataFields.map((dataField) => (
                    <TableRow key={dataField.id}>
                      <TableCell className="font-medium">{dataField.name}</TableCell>
                      <TableCell>{dataField.category}</TableCell>
                      <TableCell>{getStatusBadge(dataField.status)}</TableCell>
                      <TableCell>{dataField.vectorCount}</TableCell>
                      <TableCell>
                        {dataField.lastUploaded 
                          ? new Date(dataField.lastUploaded).toLocaleString() 
                          : 'Never'}
                      </TableCell>
                      <TableCell>
                        {dataField.status !== 'Uploaded' && (
                          <Button 
                            size="sm" 
                            onClick={() => handleUploadDataField(dataField.id)}
                            disabled={dataField.status === 'In Progress' || !!uploadProgress[dataField.id]}
                          >
                            <Upload className="w-4 h-4 mr-1" />
                            Upload
                          </Button>
                        )}
                        {uploadProgress[dataField.id] !== undefined && uploadProgress[dataField.id] < 100 && (
                          <Progress value={uploadProgress[dataField.id]} className="w-20" />
                        )}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="indexes" className="space-y-4">
          <div className="flex justify-between items-center">
            <h3 className="text-lg font-medium">Pinecone Indexes</h3>
            <Dialog open={isCreateIndexDialogOpen} onOpenChange={setIsCreateIndexDialogOpen}>
              <DialogTrigger asChild>
                <Button>
                  <Plus className="w-4 h-4 mr-2" />
                  Create Index
                </Button>
              </DialogTrigger>
              <DialogContent className="sm:max-w-[500px]">
                <DialogHeader>
                  <DialogTitle>Create New Index</DialogTitle>
                  <DialogDescription>
                    Create a new Pinecone index for vector storage.
                  </DialogDescription>
                </DialogHeader>
                <div className="grid gap-4 py-4">
                  <div className="grid grid-cols-4 items-center gap-4">
                    <Label htmlFor="index-name" className="text-right">
                      Name
                    </Label>
                    <Input
                      id="index-name"
                      value={newIndexName}
                      onChange={(e) => setNewIndexName(e.target.value)}
                      className="col-span-3"
                      placeholder="worldquant-miner-prod"
                    />
                  </div>
                  <div className="grid grid-cols-4 items-center gap-4">
                    <Label htmlFor="index-dimension" className="text-right">
                      Dimension
                    </Label>
                    <Input
                      id="index-dimension"
                      type="number"
                      value={newIndexDimension}
                      onChange={(e) => setNewIndexDimension(e.target.value)}
                      className="col-span-3"
                      placeholder="1536"
                    />
                  </div>
                  <div className="grid grid-cols-4 items-center gap-4">
                    <Label htmlFor="index-metric" className="text-right">
                      Metric
                    </Label>
                    <Select value={newIndexMetric} onValueChange={setNewIndexMetric}>
                      <SelectTrigger className="col-span-3">
                        <SelectValue placeholder="Select metric" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="cosine">Cosine</SelectItem>
                        <SelectItem value="dotproduct">Dot Product</SelectItem>
                        <SelectItem value="euclidean">Euclidean</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="grid grid-cols-4 items-center gap-4">
                    <Label htmlFor="deletion-protection" className="text-right">
                      Deletion Protection
                    </Label>
                    <div className="col-span-3 flex items-center space-x-2">
                      <Switch
                        id="deletion-protection"
                        checked={newIndexDeletionProtection}
                        onCheckedChange={setNewIndexDeletionProtection}
                      />
                      <Label htmlFor="deletion-protection">Enable</Label>
                    </div>
                  </div>
                  <div className="grid grid-cols-4 items-center gap-4">
                    <Label className="text-right">
                      Tags
                    </Label>
                    <div className="col-span-3 space-y-2">
                      <div className="flex space-x-2">
                        <Input
                          placeholder="Key"
                          value={newTagKey}
                          onChange={(e) => setNewTagKey(e.target.value)}
                        />
                        <Input
                          placeholder="Value"
                          value={newTagValue}
                          onChange={(e) => setNewTagValue(e.target.value)}
                        />
                        <Button type="button" onClick={handleAddNewTag}>
                          <Plus className="w-4 h-4" />
                        </Button>
                      </div>
                      <div className="flex flex-wrap gap-2 mt-2">
                        {Object.entries(newIndexTags).map(([key, value]) => (
                          <Badge key={key} variant="outline" className="flex items-center">
                            {key}: {value}
                            <Button
                              variant="ghost"
                              size="sm"
                              className="h-4 w-4 p-0 ml-1"
                              onClick={() => handleRemoveNewTag(key)}
                            >
                              <XCircle className="h-3 w-3" />
                            </Button>
                          </Badge>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
                <DialogFooter>
                  <Button variant="outline" onClick={() => setIsCreateIndexDialogOpen(false)}>
                    Cancel
                  </Button>
                  <Button onClick={handleCreateIndex} disabled={isLoading}>
                    {isLoading ? (
                      <>
                        <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                        Creating...
                      </>
                    ) : (
                      'Create Index'
                    )}
                  </Button>
                </DialogFooter>
              </DialogContent>
            </Dialog>
          </div>

          <Card>
            <CardHeader>
              <CardTitle>Indexes</CardTitle>
              <CardDescription>Vector database indexes</CardDescription>
            </CardHeader>
            <CardContent>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Name</TableHead>
                    <TableHead>Dimension</TableHead>
                    <TableHead>Metric</TableHead>
                    <TableHead>Status</TableHead>
                    <TableHead>Vector Count</TableHead>
                    <TableHead>Last Updated</TableHead>
                    <TableHead>Actions</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {indexes.map((index) => (
                    <TableRow key={index.id}>
                      <TableCell className="font-medium">{index.id}</TableCell>
                      <TableCell>{index.dimension}</TableCell>
                      <TableCell>{index.metric}</TableCell>
                      <TableCell>
                        <Badge variant={index.status === 'Ready' ? 'default' : index.status === 'Initializing' ? 'secondary' : 'outline'}>
                          {index.status}
                        </Badge>
                      </TableCell>
                      <TableCell>{index.vectorCount}</TableCell>
                      <TableCell>{new Date(index.lastUpdated).toLocaleString()}</TableCell>
                      <TableCell>
                        <div className="flex space-x-2">
                          <Button
                            size="sm"
                            variant="outline"
                            onClick={() => handleViewIndexDetails(index)}
                          >
                            <Database className="w-4 h-4" />
                          </Button>
                          <Button
                            size="sm"
                            variant="outline"
                            onClick={() => {
                              handleSelectIndex(index);
                              setIsConfigureIndexDialogOpen(true);
                            }}
                          >
                            <Settings className="w-4 h-4" />
                          </Button>
                          <Button
                            size="sm"
                            variant="destructive"
                            onClick={() => {
                              handleSelectIndex(index);
                              setIsDeleteIndexDialogOpen(true);
                            }}
                            disabled={index.deletion_protection === 'enabled'}
                          >
                            <Trash2 className="w-4 h-4" />
                          </Button>
                        </div>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {/* Configure Index Dialog */}
      <Dialog open={isConfigureIndexDialogOpen} onOpenChange={setIsConfigureIndexDialogOpen}>
        <DialogContent className="sm:max-w-[500px]">
          <DialogHeader>
            <DialogTitle>Configure Index: {selectedIndex?.id}</DialogTitle>
            <DialogDescription>
              Configure settings for this Pinecone index.
            </DialogDescription>
          </DialogHeader>
          <div className="grid gap-4 py-4">
            <div className="grid grid-cols-4 items-center gap-4">
              <Label htmlFor="configure-deletion-protection" className="text-right">
                Deletion Protection
              </Label>
              <div className="col-span-3 flex items-center space-x-2">
                <Switch
                  id="configure-deletion-protection"
                  checked={configureDeletionProtection}
                  onCheckedChange={setConfigureDeletionProtection}
                />
                <Label htmlFor="configure-deletion-protection">Enable</Label>
              </div>
            </div>
            <div className="grid grid-cols-4 items-center gap-4">
              <Label className="text-right">
                Tags
              </Label>
              <div className="col-span-3 space-y-2">
                <div className="flex space-x-2">
                  <Input
                    placeholder="Key"
                    value={configureTagKey}
                    onChange={(e) => setConfigureTagKey(e.target.value)}
                  />
                  <Input
                    placeholder="Value"
                    value={configureTagValue}
                    onChange={(e) => setConfigureTagValue(e.target.value)}
                  />
                  <Button type="button" onClick={handleAddConfigureTag}>
                    <Plus className="w-4 h-4" />
                  </Button>
                </div>
                <div className="flex flex-wrap gap-2 mt-2">
                  {Object.entries(configureTags).map(([key, value]) => (
                    <Badge key={key} variant="outline" className="flex items-center">
                      {key}: {value}
                      <Button
                        variant="ghost"
                        size="sm"
                        className="h-4 w-4 p-0 ml-1"
                        onClick={() => handleRemoveConfigureTag(key)}
                      >
                        <XCircle className="h-3 w-3" />
                      </Button>
                    </Badge>
                  ))}
                </div>
              </div>
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setIsConfigureIndexDialogOpen(false)}>
              Cancel
            </Button>
            <Button onClick={handleConfigureIndex} disabled={isLoading}>
              {isLoading ? (
                <>
                  <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                  Saving...
                </>
              ) : (
                'Save Changes'
              )}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Delete Index Dialog */}
      <Dialog open={isDeleteIndexDialogOpen} onOpenChange={setIsDeleteIndexDialogOpen}>
        <DialogContent className="sm:max-w-[425px]">
          <DialogHeader>
            <DialogTitle>Delete Index</DialogTitle>
            <DialogDescription>
              Are you sure you want to delete the index "{selectedIndex?.id}"? This action cannot be undone.
            </DialogDescription>
          </DialogHeader>
          <div className="py-4">
            {selectedIndex?.deletion_protection === 'enabled' && (
              <Alert variant="destructive" className="mb-4">
                <AlertCircle className="h-4 w-4" />
                <AlertTitle>Deletion Protection Enabled</AlertTitle>
                <AlertDescription>
                  This index has deletion protection enabled. You must disable it before deleting the index.
                </AlertDescription>
              </Alert>
            )}
            <div className="space-y-2">
              <p><strong>Name:</strong> {selectedIndex?.id}</p>
              <p><strong>Dimension:</strong> {selectedIndex?.dimension}</p>
              <p><strong>Vector Count:</strong> {selectedIndex?.vectorCount}</p>
              <p><strong>Status:</strong> {selectedIndex?.status}</p>
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setIsDeleteIndexDialogOpen(false)}>
              Cancel
            </Button>
            <Button 
              variant="destructive" 
              onClick={handleDeleteIndex} 
              disabled={isLoading || selectedIndex?.deletion_protection === 'enabled'}
            >
              {isLoading ? (
                <>
                  <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                  Deleting...
                </>
              ) : (
                <>
                  <Trash2 className="w-4 h-4 mr-2" />
                  Delete Index
                </>
              )}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Index Details Dialog */}
      <Dialog open={isIndexDetailsOpen} onOpenChange={setIsIndexDetailsOpen}>
        <DialogContent className="sm:max-w-[600px]">
          <DialogHeader>
            <DialogTitle>Index Details: {selectedIndexDetails?.id}</DialogTitle>
            <DialogDescription>
              Detailed information about this Pinecone index
            </DialogDescription>
          </DialogHeader>
          <div className="grid gap-4 py-4">
            {selectedIndexDetails && (
              <div className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <h4 className="text-sm font-medium">Name</h4>
                    <p className="text-sm">{selectedIndexDetails.id}</p>
                  </div>
                  <div>
                    <h4 className="text-sm font-medium">Status</h4>
                    <Badge variant={selectedIndexDetails.status === 'Ready' ? 'default' : selectedIndexDetails.status === 'Initializing' ? 'secondary' : 'outline'}>
                      {selectedIndexDetails.status}
                    </Badge>
                  </div>
                  <div>
                    <h4 className="text-sm font-medium">Dimension</h4>
                    <p className="text-sm">{selectedIndexDetails.dimension}</p>
                  </div>
                  <div>
                    <h4 className="text-sm font-medium">Metric</h4>
                    <p className="text-sm">{selectedIndexDetails.metric}</p>
                  </div>
                  <div>
                    <h4 className="text-sm font-medium">Vector Count</h4>
                    <p className="text-sm">{selectedIndexDetails.vectorCount.toLocaleString()}</p>
                  </div>
                  <div>
                    <h4 className="text-sm font-medium">Last Updated</h4>
                    <p className="text-sm">{new Date(selectedIndexDetails.lastUpdated).toLocaleString()}</p>
                  </div>
                  <div>
                    <h4 className="text-sm font-medium">Deletion Protection</h4>
                    <p className="text-sm">{selectedIndexDetails.deletion_protection === 'enabled' ? 'Enabled' : 'Disabled'}</p>
                  </div>
                  <div>
                    <h4 className="text-sm font-medium">Vector Type</h4>
                    <p className="text-sm">{selectedIndexDetails.vector_type || 'dense'}</p>
                  </div>
                </div>
                
                {selectedIndexDetails.host && (
                  <div>
                    <h4 className="text-sm font-medium">Host</h4>
                    <p className="text-sm font-mono text-xs break-all">{selectedIndexDetails.host}</p>
                  </div>
                )}
                
                {selectedIndexDetails.tags && Object.keys(selectedIndexDetails.tags).length > 0 && (
                  <div>
                    <h4 className="text-sm font-medium">Tags</h4>
                    <div className="flex flex-wrap gap-2 mt-1">
                      {Object.entries(selectedIndexDetails.tags).map(([key, value]) => (
                        <Badge key={key} variant="outline">
                          {key}: {value}
                        </Badge>
                      ))}
                    </div>
                  </div>
                )}
                
                {selectedIndexDetails.spec?.serverless && (
                  <div>
                    <h4 className="text-sm font-medium">Serverless Configuration</h4>
                    <div className="grid grid-cols-2 gap-4 mt-1">
                      <div>
                        <h5 className="text-xs font-medium">Cloud</h5>
                        <p className="text-sm">{selectedIndexDetails.spec.serverless.cloud || 'N/A'}</p>
                      </div>
                      <div>
                        <h5 className="text-xs font-medium">Region</h5>
                        <p className="text-sm">{selectedIndexDetails.spec.serverless.region || 'N/A'}</p>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setIsIndexDetailsOpen(false)}>
              Close
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
} 