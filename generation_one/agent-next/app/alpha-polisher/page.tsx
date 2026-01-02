'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { getStoredJWT } from '../../lib/auth';
import { createChatCompletion } from '../../lib/deepseek';
import OperatorSelector from '../../components/OperatorSelector';
import { Button } from '../../components/ui/button';
import { Textarea } from '../../components/ui/textarea';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '../../components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../../components/ui/tabs';
import { Loader2, Check, X, Play } from 'lucide-react';
import { FloatingDock } from '../../components/ui/floating-dock';
import { sharedNavItems } from '../../components/ui/shared-navigation';
import { openDatabase, addSimulation } from '@/lib/indexedDB';
import { toast } from 'sonner';

// Constants for token limits
const MAX_TOKENS = 4000; // Approximate token limit for DeepSeek
const TOKENS_PER_CHAR = 0.25; // Approximate tokens per character

// Add type definitions
interface PredictedMetrics {
  sharpe: string;
  fitness: string;
  return: string;
  drawdown: string;
}

interface AlphaIdea {
  improved_expression: string;
  reasoning: string;
  predicted_metrics: PredictedMetrics;
  suggested_operators: string[];
}

interface AlphaResponse {
  ideas: AlphaIdea[];
}

interface Operator {
  name: string;
  category: string;
  scope: string[];
  definition: string;
  description: string;
  documentation: string;
  level: string;
  requiresLogin?: boolean;
}

interface Simulation {
  id: string;
  alpha_expression: string;
  status: 'queued' | 'simulating' | 'completed' | 'error';
  progress: number;
  result?: {
    fitness: number;
    sharpe: number;
    turnover: number;
    [key: string]: any;
  };
  error?: string;
  created_at: number;
  updated_at: number;
}

export default function AlphaPolisherPage() {
  const router = useRouter();
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [selectedOperators, setSelectedOperators] = useState<string[]>([]);
  const [expression, setExpression] = useState('');
  const [suggestions, setSuggestions] = useState('');
  const [generatedIdeas, setGeneratedIdeas] = useState<(AlphaIdea | string)[]>([]);
  const [isGenerating, setIsGenerating] = useState(false);
  const [activeTab, setActiveTab] = useState('expression');
  const [isMounted, setIsMounted] = useState(false);
  const [allOperators, setAllOperators] = useState<Operator[]>([]);
  const [currentSharpe, setCurrentSharpe] = useState('');
  const [currentFitness, setCurrentFitness] = useState('');
  const [currentReturn, setCurrentReturn] = useState('');
  const [currentDrawdown, setCurrentDrawdown] = useState('');
  const [streamingResponse, setStreamingResponse] = useState('');
  const [requiresLogin, setRequiresLogin] = useState(false);
  const [queuedSimulations, setQueuedSimulations] = useState<Set<string>>(new Set());

  useEffect(() => {
    setIsMounted(true);
    const checkAuth = async () => {
      const jwtToken = getStoredJWT();
      if (!jwtToken) {
        router.push('/login');
        return;
      }
      setIsAuthenticated(true);
      setIsLoading(false);
    };

    checkAuth();
  }, [router]);

  // Fetch all available operators and check login requirements on component mount
  useEffect(() => {
    const fetchOperatorsAndCheckLogin = async () => {
      try {
        const jwtToken = getStoredJWT();
        
        if (!jwtToken) {
          router.push('/login');
          return;
        }
        
        const operatorsResponse = await fetch('/api/operators', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            jwtToken,
          }),
        });
        
        if (!operatorsResponse.ok) {
          console.error('Failed to fetch operators:', operatorsResponse.statusText);
          router.push('/login');
          return;
        }
        
        const operatorsData = await operatorsResponse.json();
        setAllOperators(operatorsData);
        
        // Select all operators by default
        const operatorNames = operatorsData.map((op: Operator) => op.name);
        setSelectedOperators(operatorNames);
        
        const requiresLogin = operatorsData.some((op: Operator) => op.requiresLogin);
        setRequiresLogin(requiresLogin);
        
        if (requiresLogin && !isAuthenticated) {
          router.push('/login');
        }
      } catch (error) {
        console.error('Error fetching operators:', error);
        router.push('/login');
      }
    };

    if (isAuthenticated) {
      fetchOperatorsAndCheckLogin();
    }
  }, [isAuthenticated, router]);

  // Load saved suggestions on mount
  useEffect(() => {
    if (typeof window !== 'undefined') {
      const savedExpression = localStorage.getItem('alpha-polisher-expression');
      const savedSuggestions = localStorage.getItem('alpha-polisher-suggestions');
      if (savedExpression) setExpression(savedExpression);
      if (savedSuggestions) setSuggestions(savedSuggestions);
    }
  }, []);

  // Save suggestions when they change
  useEffect(() => {
    if (typeof window !== 'undefined') {
      localStorage.setItem('alpha-polisher-expression', expression);
      localStorage.setItem('alpha-polisher-suggestions', suggestions);
    }
  }, [expression, suggestions]);

  const handleOperatorsSelected = (operators: string[]) => {
    setSelectedOperators(operators);
  };

  // Handle select all operators
  const handleSelectAllOperators = () => {
    setSelectedOperators(allOperators.map(op => op.name));
  };

  // Handle deselect all operators
  const handleDeselectAllOperators = () => {
    setSelectedOperators([]);
  };

  // Estimate token count for the prompt
  const estimateTokenCount = (text: string): number => {
    return Math.ceil(text.length * TOKENS_PER_CHAR);
  };

  // Implement RAG (Retrieval-Augmented Generation) for large inputs
  const implementRAG = async (prompt: string): Promise<string> => {
    // This is a simplified RAG implementation
    // In a real application, you would:
    // 1. Split the prompt into chunks
    // 2. Process each chunk separately
    // 3. Combine the results
    
    // For now, we'll just split the prompt into smaller parts
    const chunks = splitIntoChunks(prompt, MAX_TOKENS);
    const results: string[] = [];
    
    for (const chunk of chunks) {
      const response = await createChatCompletion([
        { role: 'system', content: 'You are an expert in quantitative finance and alpha generation for the WorldQuant platform.' },
        { role: 'user', content: chunk }
      ]);
      results.push(response);
    }
    
    // Combine the results
    return results.join('\n\n');
  };

  // Split text into chunks based on token limit
  const splitIntoChunks = (text: string, maxTokens: number): string[] => {
    const chunks: string[] = [];
    const maxChars = Math.floor(maxTokens / TOKENS_PER_CHAR);
    
    // Simple splitting by paragraphs
    const paragraphs = text.split('\n\n');
    let currentChunk = '';
    
    for (const paragraph of paragraphs) {
      if (estimateTokenCount(currentChunk + paragraph) > maxTokens) {
        if (currentChunk) {
          chunks.push(currentChunk);
          currentChunk = paragraph;
        } else {
          // If a single paragraph exceeds the limit, split it by sentences
          const sentences = paragraph.split('. ');
          currentChunk = '';
          
          for (const sentence of sentences) {
            if (estimateTokenCount(currentChunk + sentence) > maxTokens) {
              if (currentChunk) {
                chunks.push(currentChunk);
                currentChunk = sentence;
              } else {
                // If a single sentence exceeds the limit, just add it
                chunks.push(sentence);
                currentChunk = '';
              }
            } else {
              currentChunk += (currentChunk ? '. ' : '') + sentence;
            }
          }
        }
      } else {
        currentChunk += (currentChunk ? '\n\n' : '') + paragraph;
      }
    }
    
    if (currentChunk) {
      chunks.push(currentChunk);
    }
    
    return chunks;
  };

  const generateAlphaIdeas = async () => {
    if (!expression && !suggestions) {
      alert('Please provide an expression or suggestions');
      return;
    }

    setIsGenerating(true);
    setStreamingResponse('');
    
    try {
      const systemPrompt = `You are an expert in quantitative finance and alpha generation for the WorldQuant platform. 
Your task is to analyze the provided alpha expression and its performance metrics, then suggest improvements.
Focus on creating alphas that are likely to be profitable, statistically significant, and economically sound.
Consider the current performance metrics and suggest specific improvements that could enhance them.

IMPORTANT:
1. Only use the operators provided in the selected operators list. Do not add any other operators.
2. Do not include any Python code or programming syntax.
3. Format your response as a valid JSON object with the following structure:
{
  "ideas": [
    {
      "improved_expression": "string",
      "reasoning": "string",
      "predicted_metrics": {
        "sharpe": "string",
        "fitness": "string",
        "return": "string",
        "drawdown": "string"
      },
      "suggested_operators": ["string"]
    }
  ]
}
4. Keep the alpha expressions simple and clear, using only the provided operators.
5. Use the operator definitions provided to ensure correct usage.`;

      // Format operator information for the prompt
      const operatorInfo = allOperators
        .filter(op => selectedOperators.includes(op.name))
        .map(op => `Operator: ${op.name}
Definition: ${op.definition}
Description: ${op.description}
Category: ${op.category}
Scope: ${op.scope.join(', ')}`)
        .join('\n\n');

      let userPrompt = '';
      
      if (activeTab === 'expression') {
        userPrompt = `I have an alpha expression: "${expression}"
        
Current Performance Metrics:
- Sharpe Ratio: ${currentSharpe || 'Not provided'}
- Fitness: ${currentFitness || 'Not provided'}
- Return: ${currentReturn || 'Not provided'}
- Maximum Drawdown: ${currentDrawdown || 'Not provided'}

Available Operators and their definitions:
${operatorInfo}

Please analyze this expression and its performance metrics, then suggest 5 specific improvements or variations that might improve its performance. 
For each suggestion, provide:
1. The improved alpha expression (using only the provided operators)
2. The reasoning behind the improvement
3. Predicted impact on performance metrics
4. Additional operators from the provided list that could be beneficial`;

      } else {
        userPrompt = `I want to create alpha ideas based on these suggestions: "${suggestions}"
        
Available Operators and their definitions:
${operatorInfo}

Please generate 5 alpha expressions based on these suggestions, using only the selected operators. 
For each alpha, provide:
1. The complete alpha expression (using only the provided operators)
2. The reasoning behind it
3. Predicted performance metrics
4. Additional operators from the provided list that could be beneficial`;
      }

      // Use streaming API with deep-reasoner
      const response = await fetch('/api/deepseek-stream', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          messages: [
            { role: 'system', content: systemPrompt },
            { role: 'user', content: userPrompt }
          ],
          model: 'deepseek-reasoner',
          stream: true
        })
      });

      if (!response.ok) {
        throw new Error('Failed to generate alpha ideas');
      }

      const reader = response.body?.getReader();
      const decoder = new TextDecoder();
      let accumulatedResponse = '';

      while (true) {
        const { done, value } = await reader!.read();
        if (done) break;
        
        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');
        
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.slice(6);
            if (data === '[DONE]') break;
            
            try {
              const parsed = JSON.parse(data);
              if (parsed.choices?.[0]?.delta?.content) {
                accumulatedResponse += parsed.choices[0].delta.content;
                setStreamingResponse(accumulatedResponse);
              }
            } catch (e) {
              console.error('Error parsing streaming response:', e);
            }
          }
        }
      }

      try {
        // Clean the response by removing markdown formatting
        let cleanResponse = accumulatedResponse
          .replace(/```json\n/g, '')  // Remove opening ```json
          .replace(/```\n/g, '')      // Remove closing ```
          .replace(/```/g, '')        // Remove any remaining ```
          .trim();                    // Remove extra whitespace

        // Parse the final JSON response
        const parsedResponse = JSON.parse(cleanResponse) as AlphaResponse;
        if (parsedResponse.ideas) {
          setGeneratedIdeas(parsedResponse.ideas);
        }
      } catch (e) {
        console.error('Error parsing final response:', e);
        console.log('Raw response:', accumulatedResponse);
        setGeneratedIdeas([accumulatedResponse]);
      }
    } catch (error) {
      console.error('Error generating alpha ideas:', error);
      alert('Failed to generate alpha ideas. Please try again.');
    } finally {
      setIsGenerating(false);
    }
  };

  const handleQueueSimulation = async (alpha: string) => {
    try {
      const db = await openDatabase();
      
      const simulation: Simulation = {
        id: Date.now().toString(),
        alpha_expression: alpha,
        status: 'queued',
        progress: 0,
        created_at: Date.now(),
        updated_at: Date.now()
      };
      
      await addSimulation(db, simulation);
      setQueuedSimulations(prev => new Set([...prev, alpha]));
      toast.success('Alpha queued for simulation');
    } catch (error) {
      console.error('Error queueing simulation:', error);
      toast.error('Failed to queue simulation');
    }
  };

  const handleClearData = () => {
    setExpression('');
    setSuggestions('');
    setGeneratedIdeas([]);
    setStreamingResponse('');
    if (typeof window !== 'undefined') {
      localStorage.removeItem('alpha-polisher-expression');
      localStorage.removeItem('alpha-polisher-suggestions');
    }
    toast.success('Data cleared');
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <Loader2 className="h-8 w-8 animate-spin" />
      </div>
    );
  }

  if (!isAuthenticated) {
    return null; // Will redirect to login
  }

  return (
    <div className="container mx-auto py-8">
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-3xl font-bold">Alpha Polisher</h1>
        <Button
          variant="outline"
          size="sm"
          onClick={handleClearData}
          className="flex items-center gap-2"
        >
          <X className="h-4 w-4" />
          Clear Data
        </Button>
      </div>
      <p className="text-muted-foreground mb-8">
        Use this tool to polish your alpha expressions or generate new ideas based on your suggestions.
      </p>

      <div className="grid grid-cols-1 gap-8 mb-8">
        <Card>
          <CardHeader>
            <CardTitle>Operators</CardTitle>
            <CardDescription>Select the operators you want to use in your alpha</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="mb-4 flex space-x-2">
              <Button 
                variant="outline" 
                size="sm" 
                onClick={handleSelectAllOperators}
                className="flex items-center"
              >
                <Check className="h-4 w-4 mr-1" />
                Select All
              </Button>
              <Button 
                variant="outline" 
                size="sm" 
                onClick={handleDeselectAllOperators}
                className="flex items-center"
              >
                <X className="h-4 w-4 mr-1" />
                Deselect All
              </Button>
            </div>
            <OperatorSelector 
              onOperatorsSelected={handleOperatorsSelected} 
              selectedOperators={selectedOperators}
            />
          </CardContent>
        </Card>
      </div>

      <Card className="mb-8">
        <CardHeader>
          <CardTitle>Alpha Input</CardTitle>
          <CardDescription>Enter your alpha expression or suggestions</CardDescription>
        </CardHeader>
        <CardContent>
          <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
            <TabsList className="grid w-full grid-cols-2">
              <TabsTrigger value="expression">Expression</TabsTrigger>
              <TabsTrigger value="suggestions">Suggestions</TabsTrigger>
            </TabsList>
            <TabsContent value="expression">
              <div className="space-y-4">
                <Textarea
                  placeholder="Enter your alpha expression here..."
                  className="min-h-[150px]"
                  value={expression}
                  onChange={(e) => setExpression(e.target.value)}
                />
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Current Sharpe Ratio</label>
                    <input
                      type="number"
                      className="w-full p-2 border rounded"
                      value={currentSharpe}
                      onChange={(e) => setCurrentSharpe(e.target.value)}
                      placeholder="Enter current Sharpe ratio"
                    />
                  </div>
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Current Fitness</label>
                    <input
                      type="number"
                      className="w-full p-2 border rounded"
                      value={currentFitness}
                      onChange={(e) => setCurrentFitness(e.target.value)}
                      placeholder="Enter current fitness"
                    />
                  </div>
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Current Return (%)</label>
                    <input
                      type="number"
                      className="w-full p-2 border rounded"
                      value={currentReturn}
                      onChange={(e) => setCurrentReturn(e.target.value)}
                      placeholder="Enter current return"
                    />
                  </div>
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Maximum Drawdown (%)</label>
                    <input
                      type="number"
                      className="w-full p-2 border rounded"
                      value={currentDrawdown}
                      onChange={(e) => setCurrentDrawdown(e.target.value)}
                      placeholder="Enter maximum drawdown"
                    />
                  </div>
                </div>
              </div>
            </TabsContent>
            <TabsContent value="suggestions">
              <Textarea
                placeholder="Enter your suggestions for alpha ideas..."
                className="min-h-[150px]"
                value={suggestions}
                onChange={(e) => setSuggestions(e.target.value)}
              />
            </TabsContent>
          </Tabs>
        </CardContent>
        <CardFooter>
          <Button 
            onClick={generateAlphaIdeas} 
            disabled={isGenerating || (!expression && !suggestions)}
            className="w-full"
          >
            {isGenerating ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Generating...
              </>
            ) : (
              'Generate Alpha Ideas'
            )}
          </Button>
        </CardFooter>
      </Card>

      {streamingResponse && (
        <Card className="mb-8">
          <CardHeader>
            <CardTitle>Streaming Response</CardTitle>
            <CardDescription>Real-time generation of alpha ideas</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="prose max-w-none dark:prose-invert">
              <pre className="whitespace-pre-wrap">{streamingResponse}</pre>
            </div>
          </CardContent>
        </Card>
      )}

      {generatedIdeas.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Generated Alpha Ideas</CardTitle>
            <CardDescription>Here are the final alpha ideas based on your input</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-6">
              {generatedIdeas.map((idea, index) => (
                <div key={index} className="p-4 border rounded-lg">
                  <div className="prose max-w-none dark:prose-invert">
                    {typeof idea === 'string' ? (
                      <pre className="whitespace-pre-wrap">{idea}</pre>
                    ) : (
                      <>
                        <h3 className="font-semibold">Improved Expression:</h3>
                        <p className="font-mono bg-muted p-2 rounded">{idea.improved_expression}</p>
                        
                        <h3 className="font-semibold mt-4">Reasoning:</h3>
                        <p>{idea.reasoning}</p>
                        
                        <h3 className="font-semibold mt-4">Predicted Metrics:</h3>
                        <ul>
                          <li>Sharpe Ratio: {idea.predicted_metrics.sharpe}</li>
                          <li>Fitness: {idea.predicted_metrics.fitness}</li>
                          <li>Return: {idea.predicted_metrics.return}</li>
                          <li>Drawdown: {idea.predicted_metrics.drawdown}</li>
                        </ul>
                        
                        <h3 className="font-semibold mt-4">Suggested Operators:</h3>
                        <p>{idea.suggested_operators.join(', ')}</p>

                        <div className="mt-4">
                          <Button
                            variant={queuedSimulations.has(idea.improved_expression) ? "default" : "outline"}
                            size="sm"
                            onClick={() => handleQueueSimulation(idea.improved_expression)}
                            className="flex items-center gap-2"
                            disabled={queuedSimulations.has(idea.improved_expression)}
                          >
                            {queuedSimulations.has(idea.improved_expression) ? (
                              <>
                                <Check className="h-4 w-4" />
                                Queued for Simulation
                              </>
                            ) : (
                              <>
                                <Play className="h-4 w-4" />
                                Queue for Simulation
                              </>
                            )}
                          </Button>
                        </div>
                      </>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Floating Navigation Dock */}
      {isMounted && (
        <FloatingDock items={sharedNavItems} />
      )}
    </div>
  );
} 